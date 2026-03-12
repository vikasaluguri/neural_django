from django.db import models
import json


class TrainingSession(models.Model):
    name            = models.CharField(max_length=120, default='My Network')
    dataset         = models.CharField(max_length=40, default='xor')
    layer_sizes     = models.TextField()          # JSON list
    activations     = models.TextField()          # JSON list
    loss_function   = models.CharField(max_length=40, default='mse')
    learning_rate   = models.FloatField(default=0.01)
    epochs          = models.IntegerField(default=1000)
    weights         = models.TextField(blank=True)  # JSON — saved weights
    history_loss    = models.TextField(blank=True)  # JSON list
    history_acc     = models.TextField(blank=True)  # JSON list
    final_loss      = models.FloatField(null=True, blank=True)
    final_accuracy  = models.FloatField(null=True, blank=True)
    created_at      = models.DateTimeField(auto_now_add=True)
    trained_at      = models.DateTimeField(null=True, blank=True)

    def set_layer_sizes(self, lst):  self.layer_sizes  = json.dumps(lst)
    def get_layer_sizes(self):       return json.loads(self.layer_sizes)
    def set_activations(self, lst):  self.activations  = json.dumps(lst)
    def get_activations(self):       return json.loads(self.activations)
    def set_history(self, h):
        self.history_loss = json.dumps(h['loss'])
        self.history_acc  = json.dumps(h['accuracy'])
    def get_history(self):
        return {
            'loss':     json.loads(self.history_loss or '[]'),
            'accuracy': json.loads(self.history_acc  or '[]'),
        }

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.name} — {self.dataset} ({self.epochs} epochs)'


class Prediction(models.Model):
    session     = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='predictions')
    input_data  = models.TextField()    # JSON
    output_data = models.TextField()    # JSON
    created_at  = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Prediction for {self.session.name}'
