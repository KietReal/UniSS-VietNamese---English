class CheckpointTracker:
    
    def __init__(self):
        self.start_epoch = 1
        self.best_tts_wer = float('inf')
    
    def state_dict(self):
        return {
            "start_epoch": self.start_epoch, 
            "best_tts_wer": self.best_tts_wer
        }
        
    def load_state_dict(self, state_dict):
        self.start_epoch = state_dict["start_epoch"]
        self.best_tts_wer = state_dict["best_tts_wer"]
