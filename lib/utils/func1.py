import matlab.engine

def matlab_plot_pattern():
    eng = matlab.engine.start_matlab()
    eng.func2(nargout=0, x=64)
    eng.quit()

matlab_plot_pattern()
