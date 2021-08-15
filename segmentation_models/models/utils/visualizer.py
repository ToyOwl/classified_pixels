from visdom import Visdom

__all__ = ['VisdomVisualizer']

class Visualizer(object):

  def scalar(self, x,y, name):
    pass

  def image(self, img, name):
    pass

  def table(self, tbl, name, column1, column2):
    pass

class VisdomVisualizer(Visualizer):

  def __init__(self, port='8097', env='train'):

    super().__init__()
    self.vis = Visdom(port=port, env=env)
    self.env = env
    self.plot_windows = {}

  def scalar(self, x, y, name, trace=None):

    if not isinstance(x, list):
      x = [x]
    if not isinstance(y, list):
      y = [y]

    window = self.plot_windows.get(name, None)

    if window is not None:
      if trace is not None:
        self.vis.line(X=x, Y=y, update='append', win=window, name=trace)
      else:
        self.vis.line(X=x, Y=y, update='append', win=window)
    else:
      if trace is not None:
         self.plot_windows[name] = self.vis.line(X=x, Y=y, opts=dict(title=name, legend=[name], name=trace))
      else:
        self.plot_windows[name] = self.vis.line(X=x, Y=y, opts=dict(title=name, legend=[name]))


  def image(self, img, name):

    window = self.plot_windows.get(name, None)

    if window is not None:
      self.vis.image(img=img, win=window)
    else:
      self.plot_windows[name] = self.vis.image(img=img, opts=dict(title=name))
    
  def table(self, tbl, name, column1='term', column2='value'):

    window = self.plot_windows.get(name, None)

    tbl_str = '<table width=\"100%\">'
    tbl_str += '<tr><th>{}</th> <th>{}</th></tr>'.format(column1, column2)
    for k, v in tbl.items():
      tbl_str += '<tr><td>{}</td><td>{}</td></tr>'.format(k, v)
    tbl_str += '</table>'

    if window is not None:
      self.vis.text(tbl_str, win=window)
    else:
      self.plot_windows[name] = self.vis.text(tbl_str, opts=dict(title=name))
