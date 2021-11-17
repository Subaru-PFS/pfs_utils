import numpy as np
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch

__all__ = ["CircleHandler", "addPfiCursor"]

class CircleHandler(HandlerPatch):  # needed to put a Circle into the legend (!)
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5*(width - xdescent), 0.5*(height - ydescent)
        p = Circle(center, height/2)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

try:
    from lsst.display.matplotlib import DisplayImpl
except ImportError:
    DisplayImpl = None
    
if not hasattr(DisplayImpl, "set_format_coord"):  # old version of display_matplotlib
    def addPfiCursor(disp,  cobraGeom, mpt):
        """Add PFS specific information to an afwDisplay.Display

        Assumes display_matplotlib backend.
        N.b. this will be easier in the next release of display_matplotlib

        Parameters
        ----------
        display : `lsst.afw.display.Display`
           The display that we are using
        cobraGeom : `pandas.DataFrame`
           read from obdb's cobra_geometry table.  Uses cobra_id, center_x_mm, center_y_mm
        mpt : `pfs.utils.coordinates.transform.PfiTransform`
           an object able to tranform MCS pixels to mm

        Usage:
            disp = afwDisplay.Display(...)
            ...
            disp.mtv(...)
            addPfiCursor(disp, cobraGeom, mpt)

        Returns
        -------
        The callback function

        """

        axes = disp._impl.display.frame.axes
        if len(axes) < 1:
            print("addPfiCursor must be called after display.mtv()")
            return

        ax = axes[0]

        if ax.format_coord is None or \
           ax.format_coord.__doc__ is None or "PFI" not in ax.format_coord.__doc__:

            x_mcs, y_mcs = mpt.pfiToMcs(cobraGeom.center_x_mm, cobraGeom.center_y_mm)
            cobraId = cobraGeom.cobra_id

            def pfi_format_coord(x, y, disp_impl=disp._impl,
                                 old_format_coord=ax.format_coord):
                "PFI addition to display_matplotlib's cursor display"
                d = np.hypot(x - x_mcs, y - y_mcs)

                msg = f"cobraId: {cobraId[np.argmin(d)]:4}" + " "

                return msg + old_format_coord(x, y)

            ax.format_coord = pfi_format_coord
            
            return pfi_format_coord
else:
    def addPfiCursor(display, cobraGeom, mpt):
        """Add PFI specific information to an afwDisplay.Display display

        Parameters
        ----------
        display : `lsst.afw.display.Display`
           The display that we are using
        cobraGeom : `pandas.DataFrame`
           read from obdb's cobra_geometry table.  Uses cobra_id, center_x_mm, center_y_mm
        mpt : `pfs.utils.coordinates.transform.PfiTransform`
           an object able to tranform MCS pixels to mm

        Usage:
            disp = afwDisplay.Display(...)
            addPfiCursor(disp, cobraGeom, mpt)

        Returns
        -------
        The callback function

        display may be None to only return the callback
        """
        x_mcs, y_mcs = mpt.pfiToMcs(cobraGeom.center_x_mm, cobraGeom.center_y_mm)
        cobraId = cobraGeom.cobra_id

        def pfi_format_coord(x, y):
            "PFI addition to display_matplotlib's cursor display"
                
            d = np.hypot(x - x_mcs, y - y_mcs)
            return f"cobraId: {cobraId[np.argmin(d)]:4}"

        if display is not None:
            display.set_format_coord(pfi_format_coord)

        return pfi_format_coord
