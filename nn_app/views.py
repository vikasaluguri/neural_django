import json
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from .models import TrainingSession, Prediction
from .neural_network import NeuralNetwork, DATASETS


# ──────────────────────────────────────────────
#  Dashboard
# ──────────────────────────────────────────────

def dashboard(request):
    sessions = TrainingSession.objects.all()[:10]
    return render(request, 'nn_app/dashboard.html', {'sessions': sessions})


# ──────────────────────────────────────────────
#  Create / Configure Network
# ──────────────────────────────────────────────

def create_network(request):
    if request.method == 'POST':
        data = request.POST
        name          = data.get('name', 'My Network')
        dataset       = data.get('dataset', 'xor')
        loss_fn       = data.get('loss_function', 'mse')
        lr            = float(data.get('learning_rate', 0.01))
        epochs        = int(data.get('epochs', 1000))

        # Parse hidden layers
        hidden_raw    = data.get('hidden_layers', '4,4')
        try:
            hidden = [int(x.strip()) for x in hidden_raw.split(',') if x.strip()]
        except ValueError:
            hidden = [4, 4]

        # Activation per layer
        activations_raw = data.get('activations', '')
        if activations_raw:
            activations = [a.strip() for a in activations_raw.split(',')]
        else:
            activations = ['relu'] * len(hidden) + ['sigmoid']

        # Build layer_sizes from dataset input size
        X, Y, _, _ = DATASETS[dataset]()
        n_in  = X.shape[0]
        n_out = Y.shape[0]
        layer_sizes = [n_in] + hidden + [n_out]

        # Ensure activations matches number of layers
        n_layers = len(layer_sizes) - 1
        if len(activations) < n_layers:
            activations = activations + ['sigmoid'] * (n_layers - len(activations))
        activations = activations[:n_layers]

        session = TrainingSession(
            name=name,
            dataset=dataset,
            loss_function=loss_fn,
            learning_rate=lr,
            epochs=epochs,
        )
        session.set_layer_sizes(layer_sizes)
        session.set_activations(activations)
        session.save()
        return redirect('train_network', pk=session.pk)

    dataset_choices = [
        ('xor',     '⊕', 'XOR logic gate — classic non-linear'),
        ('and',     '∧', 'AND logic gate — linearly separable'),
        ('or',      '∨', 'OR logic gate — linearly separable'),
        ('circles', '◎', 'Concentric circles — curved boundary'),
        ('moons',   '☽', 'Interleaved moons — complex boundary'),
    ]
    return render(request, 'nn_app/create_network.html', {'dataset_choices': dataset_choices})


# ──────────────────────────────────────────────
#  Train
# ──────────────────────────────────────────────

def train_network(request, pk):
    session = get_object_or_404(TrainingSession, pk=pk)
    return render(request, 'nn_app/train.html', {'session': session})


@csrf_exempt
def api_train(request, pk):
    """AJAX endpoint — trains network and returns history."""
    session = get_object_or_404(TrainingSession, pk=pk)
    X, Y, feat_names, out_names = DATASETS[session.dataset]()

    nn = NeuralNetwork(
        layer_sizes   = session.get_layer_sizes(),
        activations   = session.get_activations(),
        loss          = session.loss_function,
        learning_rate = session.learning_rate,
    )
    history = nn.train(X, Y, epochs=session.epochs)

    session.set_history(history)
    session.final_loss     = history['loss'][-1]
    session.final_accuracy = history['accuracy'][-1]
    session.weights        = json.dumps(nn.to_dict())
    session.trained_at     = timezone.now()
    session.save()

    return JsonResponse({
        'status':   'ok',
        'history':  history,
        'final_loss': session.final_loss,
        'final_accuracy': session.final_accuracy,
        'layer_sizes': session.get_layer_sizes(),
        'activations': session.get_activations(),
    })


# ──────────────────────────────────────────────
#  Results
# ──────────────────────────────────────────────

def results(request, pk):
    session = get_object_or_404(TrainingSession, pk=pk)
    history = session.get_history()
    X, Y, feat_names, out_names = DATASETS[session.dataset]()

    # ── Sample rows ──────────────────────────────
    sample_size = min(20, X.shape[1])

    # Run predictions if the network has been trained
    preds_arr = None
    if session.weights:
        nn = NeuralNetwork.from_dict(json.loads(session.weights))
        preds_arr = nn.predict(X[:, :sample_size])

    # Build one combined list — no complex slicing needed in the template
    sample_rows = []
    for i in range(sample_size):
        target_val  = round(float(Y[0, i]), 4)
        row = {
            'idx':    i + 1,
            'inputs': [round(float(v), 4) for v in X[:, i]],
            'target': target_val,
        }
        if preds_arr is not None:
            prob      = round(float(preds_arr[0, i]), 4)
            pred_cls  = 1 if prob >= 0.5 else 0
            row['prob']    = prob
            row['pred_cls'] = pred_cls
            row['correct']  = (pred_cls == int(round(target_val)))
        sample_rows.append(row)

    has_predictions = preds_arr is not None

    # ── Layer spec table ─────────────────────────
    layer_sizes = session.get_layer_sizes()
    activations = session.get_activations()
    layer_specs = []
    for i, act in enumerate(activations):
        layer_specs.append({
            'index':      i + 1,
            'from_size':  layer_sizes[i],
            'to_size':    layer_sizes[i + 1],
            'activation': act,
        })

    # ── Chart data (downsample to 200 points) ────
    loss_hist = history.get('loss', [])
    acc_hist  = history.get('accuracy', [])
    step      = max(1, len(loss_hist) // 200)
    chart_loss = loss_hist[::step]
    chart_acc  = acc_hist[::step]
    chart_x    = list(range(0, len(loss_hist), step))

    # ── Dataset type flag ────────────────────────
    is_logic = session.dataset in ['xor', 'and', 'or']

    context = {
        'session':         session,
        'feat_names':      feat_names,
        'sample_rows':     sample_rows,
        'has_predictions': has_predictions,
        'layer_specs':     layer_specs,
        'chart_loss':      json.dumps(chart_loss),
        'chart_acc':       json.dumps(chart_acc),
        'chart_x':         json.dumps(chart_x),
        'layer_sizes_js':  json.dumps(layer_sizes),
        'activations_js':  json.dumps(activations),
        'is_logic':        is_logic,
    }
    return render(request, 'nn_app/results.html', context)


# ──────────────────────────────────────────────
#  Predict (manual input)
# ──────────────────────────────────────────────

@csrf_exempt
def api_predict(request, pk):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    session = get_object_or_404(TrainingSession, pk=pk)
    if not session.weights:
        return JsonResponse({'error': 'Network not trained yet'}, status=400)

    body  = json.loads(request.body)
    inputs = body.get('inputs', [])
    X = np.array(inputs).reshape(-1, 1)
    nn = NeuralNetwork.from_dict(json.loads(session.weights))
    prob  = float(nn.predict(X)[0, 0])
    cls   = int(prob >= 0.5)

    pred = Prediction.objects.create(
        session     = session,
        input_data  = json.dumps(inputs),
        output_data = json.dumps({'probability': prob, 'class': cls}),
    )
    return JsonResponse({'probability': round(prob, 4), 'class': cls})


# ──────────────────────────────────────────────
#  Delete Session
# ──────────────────────────────────────────────

def delete_session(request, pk):
    session = get_object_or_404(TrainingSession, pk=pk)
    session.delete()
    return redirect('dashboard')