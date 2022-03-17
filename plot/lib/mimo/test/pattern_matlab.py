import matlab.engine

def matlab_plot_pattern():
    eng = matlab.engine.start_matlab()
    eng.plot_pattern(nargout=0)
    eng.quit()

matlab_plot_pattern()
