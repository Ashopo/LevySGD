from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Callable, Iterable, List
from scipy.signal import savgol_filter

def nested_list_to_vectors(arr: List):
    return [np.array(x) for x in zip(*arr)]

def vectors_to_nested_list(arr: List):
    """ 
    Given [(x_1, y_1), (x_2, y_2)] get [[x_1, x_2], [y_1,y_]]
    Opposite of above.
    """
    ndim = len(arr[0])
    nested_list = []
    for n in range(ndim):
        nested_list.append([x[n] for x in arr])
    
    return nested_list

def apply_dimensionwise(arr: List, func: Callable, **kwargs):
    """
    Accepts a list of Iterables. Each list corresponds to data in 1 dimension.
    Returns a list of the given function's outputs taking each inner list as input. 
    """

    result = []

    for d in arr:
        result.append(func(d, **kwargs))
    
    return result

def center_to_start(arr: np.ndarray):
    """
    Recenters array of coordinates s.t. the origin is at the first coordinate.
    """
    return arr - arr[0]

def diff_space(arr: Iterable, spacing: int, shift: bool=False, padding: bool=False):
    """
    Computes np.diff with specified spacing if shift == False.
    Computes np.shift with specified spacing if shift == True.
    Automatically removes nan values created if padding is False.

    arr: np.array; data to compute on
    spacing: integer; how far apart shift/diff should calculate.
    """

    if isinstance(arr, List):
        arr = np.array(arr)
    elif not isinstance(arr, np.ndarray):
        raise ValueError(
            """
            Behaviour undefined on provided Iterable. Please provide list or np.ndarray.
            """)

    shifted = np.empty_like(arr)

    if spacing > 0:
        shifted[:spacing] = np.nan
        shifted[spacing:] = arr[:-spacing]
    elif spacing < 0:
        shifted[spacing:] = np.nan
        shifted[:spacing] = arr[-spacing:]
    else:
        shifted[:] = arr
    
    if shift:

        if padding: return shifted
        
        else: return shifted[~np.isnan(shifted)]
    
    diff = arr - shifted

    if not padding:
        diff = diff[~np.isnan(diff)]

    return diff
    

def compute_tmsd(path: List[List], tau: int, plots: bool=True):
    """
    Finds the tMSD of some given path and lookahead time tau.
    Specifically, this finds the time-averaged tMSD NOT an ensemble average.
    In ergodic systems, the 2 MSDs are equivalent; see below.
    https://www.youtube.com/watch?v=SvsW-DxrhRE

    path: list of list of coordinates in each dimension.
    tau: integer; lookahead in discrete index steps.
    """

    positions = apply_dimensionwise(path, center_to_start)
    displacements = apply_dimensionwise(positions, diff_space, spacing=tau)
    displacements = nested_list_to_vectors(displacements)
    rs = [x.dot(x) for x in displacements]
    tmsd = np.mean(rs)

    if not plots:
        return tmsd
    
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(x=np.arange(len(rs)), y=rs),
        row=1, col=1)
    fig.add_trace(
        go.Histogram(x=rs),
        row=1, col=2)
    fig.update_layout(
        autosize=False,
        width=1600,
        height=800
    )
    
    return tmsd, fig


def tmsd_analysis(path: List[List], max_tau_ratio: float=0.5):
    """
    Returns tMSD plots for some given path.
    Assumes all coordinate lists are of equal length.

    path: iterables of coordinates in each dimension.
    """

    if max_tau_ratio >= 1.0:
        raise ValueError("Max lookahead cannot exceed path length.")
    
    max_tau = int(len(path[0]) * max_tau_ratio)
    taus = np.arange(0, max_tau)
    tmsds = []

    for tau in taus:
        tmsd = compute_tmsd(path, tau, plots=False)
        tmsds.append(tmsd)

    tmsds = np.array(tmsds)
    tmsds = savgol_filter(tmsds, window_length=51, polyorder=1)

    diffusion_coef = derivative(np.log(taus[1:]), np.log(tmsds[1:]), 50)

    if len(path) == 2:
    
        fig = make_subplots(rows=2, cols=2)
        fig.add_trace(
            go.Scatter(
                x=path[0], y=path[1], 
                name='Walker Trajectory',
                text=list(np.arange(0, len(path[0]))))
        )

        fig.add_trace(
            go.Scatter(x=taus, y=tmsds, name='tMSD vs 𝜏'),
            row=1, col=2
        )
        fig['layout']['xaxis2']['title'] = '𝜏'
        fig['layout']['yaxis2']['title'] = 'tMSD'

        fig.add_trace(
            go.Scatter(x=taus, y=diffusion_coef, name='Diffusion Coefficient vs Tau'),
            row=2, col=1
        )
        fig['layout']['xaxis3']['title'] = '𝜏'
        fig['layout']['yaxis3']['title'] = 'Diffusion Coefficient'

        fig.update_layout(
            autosize=False,
            width=1600,
            height=800,
        )

        return fig, taus, tmsds

    else:

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=taus, y=tmsds, name='tMSD vs 𝜏')
        )
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis_title='𝜏',
            yaxis_title='tMSD'
        )

        return fig, taus, tmsds

def emsd_analysis(paths: List[List[List]], tau: int=1):
    """
    Finds the eMSD of some given ensemble of paths.
    Specifically, this finds an ensemble average NOT time-average.

    paths: list of runs with list of list of coordinates in each dimension.
    tau: fixed lookahead time for calculating displacements.
    """

    ensemble_rs = []
    for path in paths:
        positions = apply_dimensionwise(path, center_to_start)
        displacements = apply_dimensionwise(positions, diff_space, spacing=tau)
        displacements = nested_list_to_vectors(displacements)
        rs = [x.dot(x) for x in displacements]
        ensemble_rs.append(rs)
    
    ensemble_rs = np.array(ensemble_rs)
    emsds = np.mean(ensemble_rs, axis=0)
    ts = np.arange(0, len(emsds))

    if len(path) == 2:
    
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            go.Scatter(
                x=path[0], y=path[1], 
                name='Sample Ensemble Walker Trajectory',
                text=list(np.arange(0, len(path[0]))))
        )

        fig.add_trace(
            go.Scatter(x=ts, y=emsds, name='eMSD vs t'),
            row=1, col=2
        )
        fig['layout']['xaxis2']['title'] = 't'
        fig['layout']['yaxis2']['title'] = 'eMSD'
        fig.update_layout(
            autosize=False,
            width=1600,
            height=800,
        )

        return fig, ts, emsds

    else:

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=ts, y=emsds, name='eMSD vs t')
        )
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis_title='t',
            yaxis_title='eMSD'
        )

        return fig, ts, emsds


def derivative(
    X: np.ndarray,
    y: np.ndarray,
    h: int = 1
):
    if h == 1:
        return np.diff(y)/np.diff(X)

    return diff_space(y, h)/diff_space(X, h)


def rolling_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    skip_first: bool = True
):

    if len(X) != len(y):
        raise ValueError("Length mismatch between features and targets.")
    if len(X) < window:
        raise ValueError("Window size larger than data length.")

    slopes = [np.nan]*window
    
    for i in range(window, len(X)):

        if skip_first:
            slopes.append(np.nan)
            skip_first = False 
            continue

        X_window = X[i-window:i].reshape(-1,1)
        y_window = y[i-window:i].reshape(-1,1)
        mdl = LinearRegression().fit(X_window, y_window)
        slopes.append(mdl.coef_[0][0])
    
    return np.array(slopes)


def plot(
    x: np.ndarray, 
    y: np.ndarray, 
    xaxis_title: str = None,
    yaxis_title: str = None,
    title: str = None
):
    updatemenus = [
        dict(
            buttons=list([
                dict(label='Linear x',
                    method='relayout',
                    args=[{'xaxis': {'type': 'linear',
                                     'xaxis_title': xaxis_title}},
                    ]
                ),
                dict(label='Log x',
                    method='relayout',
                    args=[{'xaxis': {'type': 'log',
                                     'xaxis_title': xaxis_title}},
                    ]
                )
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            active=0,
            x=0.1,
            xanchor="left",
            y=1.08,
            yanchor="top"
        ),
        dict(
            buttons=list([
                dict(label='Linear y',
                    method='relayout',
                    args=[{'yaxis': {'type': 'linear',
                                     'yaxis_title': yaxis_title}},
                    ]
                ),
                dict(label='Log y',
                    method='relayout',
                    args=[{'yaxis': {'type': 'log',
                                     'yaxis_title': yaxis_title}},
                    ]
                )
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            active=0,
            x=0.37,
            xanchor="left",
            y=1.08,
            yanchor="top"
        )
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y)
    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        updatemenus=updatemenus,
        title=title
    )

    return fig

def figures_to_html(figs: List[go.Figure], filename: str="dashboard.html"):

    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")

def _phi(alpha, beta):
    """ Common function. """
    return beta * np.tan(np.pi * alpha / 2.0)

def levyrandom(alpha, beta, mu=0.0, sigma=1.0, shape=()):
    """
    Generate random values sampled from an alpha-stable distribution.
    Notice that this method is "exact", in the sense that is derived
    directly from the definition of stable variable.
    It uses parametrization 0 (to get it from another parametrization, convert).
    Example:
        >>> rnd = random(1.5, 0, shape=(100,))  # parametrization 0 is implicit
        >>> par = np.array([1.5, 0.905, 0.707, 1.414])
        >>> rnd = random(*Parameters.convert(par ,'B' ,'0'), shape=(100,))  # example with convert
    """

    # loc = change_par(alpha, beta, mu, sigma, par, 0)
    if alpha == 2:
        return np.random.standard_normal(shape) * np.sqrt(2.0)

    # Fails for alpha exactly equal to 1.0
    # but works fine for alpha infinitesimally greater or lower than 1.0
    radius = 1e-15  # <<< this number is *very* small
    if np.absolute(alpha - 1.0) < radius:
        # So doing this will make almost exactly no difference at all
        alpha = 1.0 + radius

    r1 = np.random.random(shape)
    r2 = np.random.random(shape)
    pi = np.pi

    a = 1.0 - alpha
    b = r1 - 0.5
    c = a * b * pi
    e = _phi(alpha, beta)
    f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
    g = np.tan(pi * b / 2.0)
    h = np.tan(c / 2.0)
    i = 1.0 - g ** 2.0
    j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
    k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)

    return mu + sigma * k

def create_density_plot(params):

    x, y = params[0], params[1]
    colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]
    fig = ff.create_2d_density(
        x, y, colorscale=colorscale, point_size=3
    )
    
    return fig