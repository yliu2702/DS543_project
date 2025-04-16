#%%
## env: conda activate /projectnb/vkolagrp/yiliu/conda_envs/env_hw_tf
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

#%%
class SlateRecSimEnv(gym.Env):
    def __init__(self, num_candidates=10, slate_size=4, num_topics=7, alpha=0.7, beta=0.1, seed=None):
        super(SlateRecSimEnv, self).__init__()
        # 有多少文档可以选
        self.num_candidates = num_candidates
        self.slate_size = slate_size
        # 有7种文档，所以doc sets 里可能有多个同topic 的文档，也有可能某个主题没有覆盖到
        self.num_topics = num_topics
        # define reward = clicks + alpha * total_watch_time - beta * bounce_penalty
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.RandomState(seed)

        # ===== Define user & doc features =====
        self.total_click_budget = 0
        self.user_total_clicks = 0
        self.topic_success_counter = np.zeros(num_topics)
        self.topic_failure_counter = np.zeros(num_topics)
        self.interest_update_rate = 0.1
        self.interest_decay_rate = 0.05

        self.observation_space = spaces.Dict({
            "user_interest": spaces.Box(low=0.0, high=1.0, shape=(num_topics,), dtype=np.float32),
            "user_age": spaces.Discrete(100),
            "user_sex": spaces.Discrete(2),
            # 谨慎/耐心；谨慎的更依赖document_popularity, document_ratings; 耐心更接受long doc
            "user_personality": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "document_topics": spaces.Box(0, 1, shape=(num_candidates, num_topics), dtype=np.int32),
            "document_qualities": spaces.Box(0.0, 1.0, shape=(num_candidates,), dtype=np.float32),
            "document_lengths": spaces.Box(1.0, 10.0, shape=(num_candidates,), dtype=np.float32),
            "document_popularity": spaces.Box(0.0, 1.0, shape=(num_candidates,), dtype=np.float32),
            "document_ratings": spaces.Box(0.0, 1.0, shape=(num_candidates,), dtype=np.float32),
        })

        self.action_space = spaces.MultiDiscrete([num_candidates] * slate_size)
        self.reset()

    def reset(self):
        # ===== 初始化用户特征 =====
        self.user_age = int(self.rng.randint(18, 70))
        self.user_sex = int(self.rng.choice([0, 1]))
        self.user_interest = self.rng.rand(self.num_topics)
        self.user_personality = self.rng.rand(2)
        self.visit_frequency = int(self.rng.choice([1, 2, 3]))
        self.time_budget = float(self.rng.choice([2, 3, 4]))
        self.total_click_budget = int(self.visit_frequency * self.time_budget)
        self.user_total_clicks = 0
        self.topic_success_counter[:] = 0
        self.topic_failure_counter[:] = 0
        self._generate_documents()
        # for render
        self.last_slate_info = []
        return self._get_obs()

    def _generate_documents(self):
        # ===== 初始化文档特征 =====
        self.document_topics = np.eye(self.num_topics)[self.rng.choice(self.num_topics, size=self.num_candidates)]
        self.document_qualities = self.rng.uniform(0.2, 1.0, size=self.num_candidates)
        self.document_lengths = self.rng.uniform(1.0, 10.0, size=self.num_candidates)
        self.document_popularity = self.rng.uniform(0.0, 1.0, size=self.num_candidates)
        self.document_ratings = self.rng.uniform(0.0, 1.0, size=self.num_candidates)

    def _get_obs(self):
        # get state
        return {
            "user_interest": self.user_interest.astype(np.float32),
            "user_age": self.user_age,
            "user_sex": self.user_sex,
            "user_personality": self.user_personality.astype(np.float32),
            "document_topics": self.document_topics.astype(np.int32),
            "document_qualities": self.document_qualities.astype(np.float32),
            "document_lengths": self.document_lengths.astype(np.float32),
            "document_popularity": self.document_popularity.astype(np.float32),
            "document_ratings": self.document_ratings.astype(np.float32),
        }

    def step(self, action):
        clicked = 0
        total_watch_time = 0.0
        bounce_penalty = 0

        self.last_slate_info = []
        # ===== 用户选择行为模型（User Choice Model） =====
        for doc_id in action:
            if self.user_total_clicks >= self.total_click_budget or self.time_budget <= 0:
                break

            topic_vec = self.document_topics[doc_id]
            topic_id = np.argmax(topic_vec)
            quality = self.document_qualities[doc_id]
            rating = self.document_ratings[doc_id]
            popularity = self.document_popularity[doc_id]
            length = self.document_lengths[doc_id]

            topic_alignment = np.dot(self.user_interest, topic_vec)
            cautiousness = self.user_personality[0]
            patience = self.user_personality[1]

            rating_pop_adjust = 1 + cautiousness * (rating + popularity)
            length_penalty = np.exp(- length / (0.1 + patience))
            # 概率模型 P(click) = topic & interest alignment * quality * (谨慎的人参考rating + popularity) * (不耐心的人愿意看短视频)
            click_prob = topic_alignment * quality * rating_pop_adjust * length_penalty
            click_prob = min(click_prob, 1.0)

            clicked_flag = False
            watch_time = 0.0
            bounced = False

            # ===== 用户点击决策后状态演化（User Transition） =====
            if self.rng.rand() < click_prob:
                clicked += 1
                clicked_flag = True
                self.user_total_clicks += 1
                # 如果doc quality > 0.7, watch_time = 1; 否则和quality 成正比
                watch_time = 1.0 if quality > 0.7 else quality
                self.time_budget -= watch_time
                total_watch_time += watch_time
                # watch_time < 0.5 记为跳出
                if watch_time < 0.5:
                    bounce_penalty += 1
                else:
                    self.topic_success_counter[topic_id] += 1
                    # 如果某topic 连续3次全看完，interest 会上涨
                    if self.topic_success_counter[topic_id] >= 3:
                        self.user_interest[topic_id] += self.interest_update_rate
                self.user_interest = np.clip(self.user_interest, 0.0, 1.0)
            else:
                self.topic_failure_counter[topic_id] += 1
                self.user_interest[topic_id] -= self.interest_decay_rate
                self.user_interest = np.clip(self.user_interest, 0.0, 1.0)
            self.last_slate_info.append({
                "doc_id": doc_id,
                "topic": topic_id,
                "quality": round(quality, 2),
                "length": round(length, 2),
                "rating": round(rating, 2),
                "popularity": round(popularity, 2),
                "clicked": clicked_flag,
                "watch_time": round(watch_time, 2),
                "bounced": bounced
            })

        self.visit_frequency -= 1
        reward = clicked + self.alpha * total_watch_time - self.beta * bounce_penalty
        if reward == 0:
            reward = 0.01

        self._generate_documents()
        done = self.visit_frequency <= 0 or self.user_total_clicks >= self.total_click_budget or self.time_budget <= 0
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print("\n=== USER STATE ===")
        print(f"User interest: {self.user_interest.round(2)} ｜ Personality: {self.user_personality.round(2)}")
        print(f"Click Budget: {self.total_click_budget} | Total Clicks: {self.user_total_clicks} | Visit left: {self.visit_frequency} | Time budget: {self.time_budget:.2f}")
        print("\n=== DOCS & USER RESPONSE FEEDBACK ===")
        for info in self.last_slate_info:
            print(f"Doc {info['doc_id']}: Topic={info['topic']}, Quality={info['quality']}, Length={info['length']}, Rating={info['rating']}, Popularity={info['popularity']}")
            print(f"Clicked={info['clicked']} | Watch Time={info['watch_time']} | Bounced={info['bounced']}")


