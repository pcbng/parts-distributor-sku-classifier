import pandas as pd
import numpy as np
import graphviz

def cross_category_bleeding(model, x_test, y_test, batch_size):
    preds = model.predict(x_test)
    cat_count = np.shape(y_test)[1]
    r = np.zeros((cat_count, cat_count))
    for x, y, pred in zip(x_test, y_test, preds):
        class_being_tested = np.argmax(y)
        class_predicted = np.argmax(pred)
        r[class_being_tested][class_predicted] += 1
    # split array into rows
    # normalize each row
    # stack rows back into one array
    return np.stack([v/np.sum(a) if v else 0 for v in a] for a in np.split(r.flatten(), cat_count))

def graphviz_cross_category_diagram(res_matrix, class_labels=[]):
    graph = graphviz.Digraph()
    label = lambda c: class_labels[c] if len(class_labels) > c else 'Category {}'.format(c)
    node_name = lambda c: 'C{}'.format(c)
    for true_c in range(len(res_matrix)):
        graph.node(node_name(true_c), label(true_c))
        preds = res_matrix[true_c]
        for pred_c in range(len(preds)):
            acc = res_matrix[true_c][pred_c] * 100.0
            if acc > 0:
                acc_str = '{:.3f}%'.format(acc)
                graph.edge(node_name(true_c), node_name(pred_c), acc_str)
    return graph

def char_by_char_classification_plot(model, partnum, x):
    x_steps = [np.array(x[:i]) for i in range(1, len(x))]
    step_preds = []
    for i, x_step in enumerate(x_steps):
        c = partnum[i] if i < len(partnum) else ''
        pred = model.predict(np.array(x_step, ndmin=2))
        step_preds.append([c] + [v for v in pred.flat])
    df_res = pd.DataFrame(step_preds, columns=['char'] + class_names)
    return df_res.plot.line(x='char', grid=True, fontsize=15, xticks=range(len(x)-1), yticks=[0.1*v for v in range(11)], figsize=(15, 5))

