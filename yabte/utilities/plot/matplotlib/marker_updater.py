class MarkerUpdater:
    """Allow updating marker sizes & alphas on zoom.

    The original is from SO by Thomas Kühn.

    https://stackoverflow.com/a/48475557/370696
    """

    def __init__(self):
        ##for storing information about Figures and Axes
        self.figs = {}

        ##for storing timers
        self.timer_dict = {}

    def add_ax(self, ax, features=[]):
        ax_dict = self.figs.setdefault(ax.figure, dict())
        ax_dict[ax] = {
            "xlim": ax.get_xlim(),
            "ylim": ax.get_ylim(),
            "figw": ax.figure.get_figwidth(),
            "figh": ax.figure.get_figheight(),
            "scale_s": 1.0,
            "scale_a": 1.0,
            "features": [features] if isinstance(features, str) else features,
        }
        ax.figure.canvas.mpl_connect("draw_event", self.update_axes)

    def update_axes(self, event):
        for fig, axes in self.figs.items():
            if fig is event.canvas.figure:
                for ax, args in axes.items():
                    ##make sure the figure is re-drawn
                    update = True

                    fw = fig.get_figwidth()
                    fh = fig.get_figheight()
                    fac1 = min(fw / args["figw"], fh / args["figh"])

                    xl = ax.get_xlim()
                    yl = ax.get_ylim()
                    fac2 = min(
                        abs(args["xlim"][1] - args["xlim"][0]) / abs(xl[1] - xl[0]),
                        abs(args["ylim"][1] - args["ylim"][0]) / abs(yl[1] - yl[0]),
                    )

                    ##factor for marker size
                    facS = (fac1 * fac2) / args["scale_s"]

                    ##factor for alpha -- limited to values smaller 1.0
                    facA = min(1.0, fac1 * fac2) / args["scale_a"]

                    ##updating the artists
                    if facS != 1.0:
                        for line in ax.lines:
                            if "size" in args["features"]:
                                line.set_markersize(line.get_markersize() * facS)

                            if "alpha" in args["features"]:
                                alpha = line.get_alpha()
                                if alpha is not None:
                                    line.set_alpha(alpha * facA)

                        for path in ax.collections:
                            if "size" in args["features"]:
                                path.set_sizes(
                                    [s * facS**2 for s in path.get_sizes()]
                                )

                            if "alpha" in args["features"]:
                                alpha = path.get_alpha()
                                if alpha is not None:
                                    path.set_alpha(alpha * facA)

                        args["scale_s"] *= facS
                        args["scale_a"] *= facA

                self._redraw_later(fig)

    def _redraw_later(self, fig):
        timer = fig.canvas.new_timer(interval=10)
        timer.single_shot = True
        timer.add_callback(lambda: fig.canvas.draw_idle())
        timer.start()

        ##stopping previous timer
        if fig in self.timer_dict:
            self.timer_dict[fig].stop()

        ##storing a reference to prevent garbage collection
        self.timer_dict[fig] = timer
