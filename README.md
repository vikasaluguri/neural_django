# 🧠 NeuralForge — Neural Network from Scratch with Django

A full-stack web application that lets you **design, train, and analyse feedforward neural networks** built entirely from scratch using **Python + NumPy** — no TensorFlow, no PyTorch, no sklearn for the core logic.

---

## 📁 Project Structure

```
neural_django/
│
├── manage.py                      ← Django entry point
├── requirements.txt
│
├── neural_project/                ← Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── nn_app/                        ← Main application
    ├── neural_network.py          ← ⭐ Core NN from scratch (NumPy only)
    ├── models.py                  ← Database models
    ├── views.py                   ← Business logic / controllers
    ├── urls.py                    ← URL routing
    ├── migrations/
    └── templates/nn_app/
        ├── base.html              ← Shared layout
        ├── dashboard.html         ← Home page
        ├── create_network.html    ← Configure network
        ├── train.html             ← Live training view
        └── results.html          ← Charts + predictions
```

---

## ⚙️ Quick Start

### 1. Prerequisites
```bash
Python >= 3.9
pip
```

### 2. Install dependencies
```bash
cd neural_django
pip install -r requirements.txt
```

### 3. Create the database
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. (Optional) Create admin account
```bash
python manage.py createsuperuser
```

### 5. Run the server
```bash
python manage.py runserver
```

### 6. Open your browser
```
http://127.0.0.1:8000/
```

---

## 🔬 How the Neural Network Works (from scratch)

### File: `nn_app/neural_network.py`

#### Activation Functions
| Name | Formula | Best for |
|------|---------|----------|
| `sigmoid` | 1 / (1 + e^-z) | Binary output |
| `relu` | max(0, z) | Hidden layers |
| `tanh` | tanh(z) | Hidden layers (zero-centred) |

#### Forward Propagation
For each layer:
```
Z[l] = W[l] · A[l-1] + b[l]
A[l] = activation(Z[l])
```

#### Loss Functions
- **MSE**: `mean((y_pred - y_true)²)`
- **Binary Cross-Entropy**: `-mean(y·log(p) + (1-y)·log(1-p))`

#### Backpropagation (Chain Rule)
```
δ[L]   = (A[L] - Y) / m                      ← output delta
dW[l]  = δ[l] · A[l-1]ᵀ                      ← weight gradient
db[l]  = sum(δ[l], axis=1)                    ← bias gradient
δ[l-1] = W[l]ᵀ · δ[l] · activation'(Z[l-1]) ← propagate delta
```

#### Weight Update (Gradient Descent)
```
W[l] := W[l] - η · dW[l]
b[l] := b[l] - η · db[l]
```

---

## 📊 Built-in Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| XOR | 4 | XOR logic gate — non-linearly separable |
| AND | 4 | AND logic gate — linearly separable |
| OR | 4 | OR logic gate — linearly separable |
| Circles | 200 | Two concentric circles |
| Moons | 200 | Two interleaved half-moons |

---

## ✅ Recommended Configurations

| Dataset | Hidden Layers | Activations | LR | Epochs |
|---------|--------------|-------------|-----|--------|
| XOR | 4, 4 | tanh, tanh, sigmoid | 0.3 | 5000 |
| AND/OR | 4 | sigmoid, sigmoid | 0.1 | 1000 |
| Circles | 8, 8, 4 | relu, relu, relu, sigmoid | 0.05 | 5000 |
| Moons | 16, 8 | tanh, tanh, sigmoid | 0.05 | 3000 |

---

## 🌐 Pages

| URL | Description |
|-----|-------------|
| `/` | Dashboard — list of training sessions |
| `/create/` | Configure a new network |
| `/train/<id>/` | Train and watch live progress |
| `/results/<id>/` | Charts, predictions, architecture |
| `/admin/` | Django admin panel |

---

## 🔌 API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/train/<id>/` | Train network, returns history JSON |
| POST | `/api/predict/<id>/` | Run inference on custom input |

---

## 🏗️ Django Concepts Used

- **Models** (`models.py`) — ORM-based database storage for sessions + predictions
- **Views** (`views.py`) — Function-based views handling GET/POST/AJAX
- **Templates** (`templates/`) — Django template language with inheritance
- **URL routing** (`urls.py`) — Named URL patterns
- **CSRF protection** — Secure AJAX calls with CSRF tokens
- **Admin** — Auto-generated admin interface

---

## 🧪 Run Tests

```bash
python manage.py test
```

---

## 📚 Learning Goals

By studying this project you will understand:

1. **Neural network math** — Forward pass, backprop, gradient descent from first principles
2. **Django MVC** — Models, Views, Templates in a real application
3. **AJAX in Django** — Async training with fetch() + JsonResponse
4. **Database design** — Storing training history and model weights as JSON
5. **NumPy vectorisation** — Efficient matrix operations for neural networks

---

## 🔮 Possible Extensions

- Add momentum / Adam optimizer
- Add dropout regularisation
- Multi-class classification (softmax output)
- Upload custom CSV datasets
- Export/import trained weights
- Add WebSockets for real-time epoch updates
