from visdom import Visdom

__all__ = ['VisdomVisualizer']

class Visualizer(object):

  def plot_func(self, x, y, name, legend):
    pass

  def plot_image(self, name, img):
    pass

  def plot_table(self, name, tbl, opts=None):
    pass

class VisdomVisualizer(Visualizer):

  def __init__(self, port='8097', env='train'):

    super().__init__()
    self.vis = Visdom(port=port, env=env)
    self.env = env
    self.plot_windows = {}

  def plot_func(self, x, y, name, legend):

    if not isinstance(x, list):
      x = [x]
    if not isinstance(y, list):
      y = [y]

    window = self.plot_windows.get(name, None)

    if window is not None:
      self.vis.line(X=x, Y=y, update='append', win=window, name=legend)
    else:
      self.plot_windows[name] = self.vis.line(X=x, Y=y, opts=dict(title=name, legend=[legend]))

  def plot_image(self, name, img):

    window = self.plot_windows.get(name, None)

    if window is not None:
      self.vis.image(img=img, win=window)
    else:
      self.plot_windows[name] = self.vis.image(img=img, opts=dict(title=name))
    
  def plot_table(self, name, tbl, opts=None):

      win = self.plot_windows.get(name, None)

      tbl_str = "<table width=\"100%\"> "
      tbl_str += "<tr> \
             <th>Term</th> \
             <th>Value</th> \
             </tr>"
      for k, v in tbl.items():
          tbl_str += "<tr> \
                   <td>%s</td> \
                   <td>%s</td> \
                   </tr>" % (k, v)

      tbl_str += "</table>"

      default_opts = {'title': name}
      if opts is not None:
          default_opts.update(opts)
      if win is not None:
          self.vis.text(tbl_str, win=win, opts=default_opts)
      else:
          self.plot_windows[name] = self.vis.text(tbl_str, opts=default_opts)
