import numpy as np
import torch


class Ensemble_PrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, state, action, reward, next_state_list, done):
        # assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        for next_state in next_state_list:
            next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state_list, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state_list, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        # print("weights",weights)
        weights /= weights.max()
        # print("weights",weights)

        weights  = np.array(weights, dtype=np.float32)
        # print("weights",weights)

        batch       = list(zip(*samples))

        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_state_lists = np.concatenate(batch[3])
        dones       = batch[4]

        return states, actions, rewards, next_state_lists, dones, indices, weights
    
    def get(self, idx):
        
        data = self.buffer[idx]
        
        state = np.concatenate(data[0])
        action = data[1]
        reward = data[2]
        next_state = np.concatenate(data[3])
        done = data[4]
        
        return state, action, reward, next_state, done
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        batch       = list(zip(*samples))

        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights
    
    def get(self, idx):
        
        data = self.buffer[idx]
        
        state = np.concatenate(data[0])
        action = data[1]
        reward = data[2]
        next_state = np.concatenate(data[3])
        done = data[4]
        
        return state, action, reward, next_state, done
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class Replay_Buffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size) 
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones
    
    def get(self, idxs, k=False):
        idxs = np.array([idxs]) 
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones
    
    def get_and_delete_from_tail(self):
        dxs = np.array([self.idx]) 
        obses = torch.as_tensor(self.obses[self.idx], device=self.device).float()
        actions = torch.as_tensor(self.actions[self.idx], device=self.device)
        rewards = torch.as_tensor(self.rewards[self.idx], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[self.idx], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[self.idx], device=self.device)
        
        # del self.obses[self.idx]
        # del self.actions[self.idx]
        # del self.rewards[self.idx]
        # del self.next_obses[self.idx]
        np.delete(self.obses, self.idx)
        np.delete(self.actions, self.idx)
        np.delete(self.rewards, self.idx)
        np.delete(self.next_obses, self.idx)
        self.idx -= 1
        
        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[5]
            self.idx = end

class Ensemble_Replay_Buffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, ensemble_num):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float64 if len(obs_shape) == 1 else np.float64

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses_list = []
        self.ensemble_number = ensemble_num
        for i in range(ensemble_num):
            self.next_obses_list.append(np.empty((capacity, *obs_shape), dtype=obs_dtype))
        self.actions = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs_list, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        for i in range(self.ensemble_number):
            np.copyto(self.next_obses_list[i][self.idx], next_obs_list[i])
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size) 
        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses_list = []
        for i in range(self.ensemble_number):
            next_obses_list.append(self.next_obses_list[i][idxs])
        dones = self.dones[idxs]
        if k:
            return obses, actions, rewards, next_obses_list, dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, rewards, next_obses_list, dones
    
    def get(self, idxs, k=False):
        idxs = np.array([idxs]) 
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses_list = []
        for i in range(self.ensemble_number):
            next_obses_list.append(torch.as_tensor(self.next_obses_list[i][idxs], device=self.device).float())
        dones = torch.as_tensor(self.dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses_list, dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, rewards, next_obses_list, dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.dones[start:end] = payload[5]
            self.idx = end


    
if __name__ == '__main__':
	pass
    
        

