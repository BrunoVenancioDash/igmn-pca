
import os, sys, csv
import numpy as np

import matplotlib
matplotlib.use('Agg')          # salva sem precisar de display (servidor/headless)
import matplotlib.pyplot as plt

try:
    here = os.path.dirname(__file__)
except NameError:
    here = os.getcwd()
sys.path.append(os.path.join(here, '..'))

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import liac


'''
Sample: Classifying MNIST digits with IGMN  (+ curva de aprendizado)
====================================================================

A IGMN aprende em PASSADA UNICA e incremental. Isso permite tracar a evolucao
das metricas conforme novos pontos chegam, sem re-treinar: a cada checkpoint
"congelamos" o estado atual e avaliamos no conjunto de teste.

Metricas tracadas vs. numero de amostras de treino vistas:
  * Acuracia (sobe)  -> fracao de classificacoes corretas (argmax do one-hot).
  * EQM      (desce) -> erro quadratico medio entre o one-hot recordado por
                        classify() e o one-hot verdadeiro, medio por dimensao.
  * Componentes      -> quantas gaussianas a IGMN criou ate aquele ponto.

Saidas: figura (PNG + PDF) e um CSV com o historico, prontos para o artigo.

Nota de dimensionalidade: covariancia CHEIA por componente custa ~O(N*K*D^3).
Com os 784 pixels crus fica inviavel/instavel (variancia zero nas bordas ->
covariancia nao positiva-definida -> erro no Cholesky). Por isso PCA antes.
'''

# ===================== CONFIG ================================================
SEED          = 42
N_CLASSES     = 10
N             = 70_000     # subamostra (IGMN e passada unica; nao precisa dos 70k)
N_COMP        = 35         # componentes do PCA (sobe acuracia, mas cuidado c/ D^3)
TEST_SIZE     = 0.3
EVAL_SUBSET   = 2000       # tamanho do teste usado NA CURVA (mantem rapido)
N_CHECKPOINTS = 22         # quantos pontos na curva (espacados em log)
CKPT_START    = 100        # primeira amostra avaliada
IGMN_DELTA    = 0.5
IGMN_TAU      = 1e-2
OUT_PREFIX    = os.path.join(here, f'igmn_mnist_curva_NCOMP{N_COMP}delta{IGMN_DELTA}_tau{IGMN_TAU}_testsize{TEST_SIZE}')
# =============================================================================
print("OUT_PREFIX: ",OUT_PREFIX)

def one_hot(label, n_classes=N_CLASSES):
    v = np.zeros(n_classes)
    v[int(label)] = 1.0
    return v


def build_checkpoints(n_train, n_points=N_CHECKPOINTS, start=CKPT_START):
    '''Indices (espacados em log) onde avaliamos durante a passada unica.
    Log -> mais densidade no inicio, onde a curva muda rapido.'''
    cps = np.logspace(np.log10(start), np.log10(n_train), n_points)
    cps = np.unique(cps.astype(int))
    cps = cps[cps >= 1]
    cps = np.append(cps, n_train)            # garante o ponto final
    return set(int(c) for c in cps)


def evaluate(igmn, Xfeat, ytrue, n_classes=N_CLASSES):
    '''Acuracia e EQM no conjunto dado. classify(feat) recorda as n_classes
    dimensoes faltantes e devolve o one-hot estimado (continuo).'''
    correct = 0
    se = 0.0
    for feat, label in zip(Xfeat, ytrue):
        out = np.asarray(igmn.classify(feat)).ravel()
        if out.shape[0] != n_classes:        # robustez: pega so a parte do rotulo
            out = out[-n_classes:]
        pred = int(np.argmax(out))
        correct += int(pred == label)
        t = one_hot(label, n_classes)
        se += float(np.sum((out - t) ** 2))
    acc = correct / len(ytrue)
    eqm = se / (len(ytrue) * n_classes)
    return acc, eqm


def plot_history(history, out_prefix=OUT_PREFIX):
    n    = np.array(history['n'])
    acc  = np.array(history['acc'])
    eqm  = np.array(history['eqm'])
    comp = np.array(history['size'])

    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.4]})

    c_acc, c_eqm = '#1f4e79', '#b22222'
    l1, = ax1.plot(n, acc, '-', color=c_acc, ms=4, lw=1.8, label='Acuracia')
    ax1.set_ylabel('Acuracia', color=c_acc)
    ax1.tick_params(axis='y', labelcolor=c_acc)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, which='both', alpha=0.3)

    ax2 = ax1.twinx()
    l2, = ax2.plot(n, eqm, '-', color=c_eqm, ms=4, lw=1.8, label='EQM')
    ax2.set_ylabel('EQM', color=c_eqm)
    ax2.tick_params(axis='y', labelcolor=c_eqm)

    ax1.set_xscale('log')
    ax1.set_title('IGMN - evolucao em passada unica (MNIST, PCA=%d)' % N_COMP)
    ax1.legend(handles=[l1, l2], loc='center right', frameon=True)

    ax3.plot(n, comp, '-', color='#2e7d32', ms=4, lw=1.6)
    ax3.set_ylabel('Componentes')
    ax3.set_xlabel('Amostras de treino vistas')
    ax3.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=200, bbox_inches='tight')
    print('figura salva em %s.png' % out_prefix)
    return fig


def save_history_csv(history, out_prefix=OUT_PREFIX):
    path = out_prefix + '.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_treino', 'acuracia', 'eqm', 'componentes'])
        for row in zip(history['n'], history['acc'],
                       history['eqm'], history['size']):
            w.writerow(row)
    print('historico salvo em %s' % path)


# ===================== DADOS =================================================
print('1. baixando MNIST...')
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32)
y = np.array(y).astype(int)
print('   ok:', X.shape, X.dtype)

rng = np.random.RandomState(SEED)
idx = rng.permutation(len(X))[:N]
X, y = X[idx], y[idx]
print('2. subamostrado para', X.shape)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

# PCA + escala ajustados SO no treino, aplicados ao teste.
pca    = PCA(n_components=N_COMP, random_state=SEED).fit(Xtr)
scaler = MinMaxScaler().fit(pca.transform(Xtr))
Xtr_r  = scaler.transform(pca.transform(Xtr))
Xte_r  = scaler.transform(pca.transform(Xte))

# subconjunto fixo do teste para a CURVA (estratificado simples por permutacao)
sub = np.random.RandomState(SEED).permutation(len(Xte_r))[:EVAL_SUBSET]
Xte_curve, yte_curve = Xte_r[sub], yte[sub]
print('3. teste completo=%d | teste-da-curva=%d' % (len(yte), len(yte_curve)))


# ===================== IGMN ==================================================
distance = np.ones(N_COMP + N_CLASSES)     # features e one-hot, ja em [0,1]
igmn = liac.models.IGMN(distance, delta=IGMN_DELTA, tau=IGMN_TAU)


# ===================== TREINO + CURVA (passada unica) ========================
checkpoints = build_checkpoints(len(Xtr_r))
history = {'n': [], 'acc': [], 'eqm': [], 'size': []}

print('4. treinando (passada unica) e avaliando nos checkpoints...')
for i, (feat, label) in enumerate(zip(Xtr_r, ytr), start=1):
    igmn.learn(np.concatenate([feat, one_hot(label)]))
    
    if i in checkpoints:
        acc, eqm = evaluate(igmn, Xte_curve, yte_curve)
        history['n'].append(i)
        history['acc'].append(acc)
        history['eqm'].append(eqm)
        history['size'].append(igmn.size)
        print('   n=%6d | acc=%.3f | eqm=%.4f | componentes=%4d'
              % (i, acc, eqm, igmn.size))
    

print('componentes criados: %d' % igmn.size)
print('dimensao:', igmn.dimension)


# ===================== AVALIACAO FINAL (teste completo) ======================
acc_full, eqm_full = evaluate(igmn, Xte_r, yte)
print('5. teste COMPLETO -> acuracia=%.3f | eqm=%.4f (%d amostras)'
      % (acc_full, eqm_full, len(yte)))


# ===================== SAIDAS ================================================
print('6. Salvando figuras...')
save_history_csv(history)
plot_history(history)