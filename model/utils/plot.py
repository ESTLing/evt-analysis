from sklearn import metrics
import plotly.graph_objects as go

def plot_roc(fig, y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    roc_auc =metrics.auc(fpr, tpr)
    print('AUC: %f' % roc_auc)

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines+markers',
        name='roc'
    ))

def plot_score(fig, scores, name):
    fig.add_trace(go.Scatter(
        x=list(range(len(scores))),
        y=scores,
        mode = 'lines',
        name=name
    ))