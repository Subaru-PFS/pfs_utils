SpectroIds:
-----------

SPS uses an ``pfs.utils.spectroIds.SpectroIds`` object to hide the
site-spectrograph-cam identification. By default this uses the
hostname that the caller is running on and a DNS site key to build the
identity. All or parts can be over-ridden by passing in, say, ``sm2`` or
``n3``. This builds and returns a dictionary of all plausible names
and numbers.

Butler:
-------

The ICS ``pfs.utils.butler.Butler`` is similar to the DRP one in some
ways:

 - it reads "map" files which contain dictionaries of object type mappings:
   ``dataMap["spsFile"] = dict(template="pfs/{pfsDay}/sps/PF{site}A{visit:06d}{spectrograph}{armNum}.fits")``
   ``configMap["cobraGeometry"] = dict(template="pfi/cobras/{moduleName}/{cobraInModule}/geometry.yaml",
                                       loaderModule="ics.cobraCharmer.cobra")``
 - those are evaluated with similar calls, e.g. ``b.get("spsFile", visit=1234)``, etc.
 - to evaluate the templates, the ``b.get()`` and ``b.put()``
   accessors use both an ``idDict`` dictionary and keyword args.

There are some differences:

 - the butler contains and creates internal keys for the templates. In
   particular, butlers are constructed with one of the above
   ``SpectroId`` objects, which provide ``site``, ``spectrograph``,
   ``cam``, ``arm``, ``armNum``, etc). And butlers provide a
   dynamically maintained ``pfsDay``.
 - the per-type dictionary is called a "tray". The only required key is ``template``.
 - there are no magic "_md" or "_filename", etc. object types. 
 - file paths are reasonable things for the ICS to want and use. If
   there is no loader key in the tray, ``b.get()`` returns the
   path. Well, ``b.getPath()`` always does, and maybe that should
   always be used.
 - For object loading, the current scheme is that if there is a
   ``loaderModule`` key in the tray, a ``.load(path)`` in that
   dynamically loaded module is called on ``b.get(objType)``. If
   instead there is a ``loader`` key, that loader object is simply
   called with the path as an argument.
 - For persisting an object, an object's ``obj.dump(path)`` is called
   on ``b.put(obj, objType)``.

 - Can dynamically set permanent keys (useful for named experiments, etc)
 - Can dynamically load map files (also useful for enginering).

Questions:
----------

- how to deal with roots (``dataRoot`` and ``configRoot``)? Explicit keys? Or per-butler properties?
- configMap vs dataMap? Useful/significant to have both? Stupid?
- Want magic loaders for YAML and/or FITS?
- Where should default ``butlerMaps.py`` go? Currently with ``butler.py``
- Allow re-evaluation? For ``reduxRoot``, say.  





