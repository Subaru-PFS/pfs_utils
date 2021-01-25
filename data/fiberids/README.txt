The columns in the grandfibermap are:
    cob: Cobra identifier (1..2394). There is one of these for each science fiber.
    fld: Field (1..3).
    cf: Cobra-in-field (1..798). cf = 57*(mf-1)+cm.
    mf: Module-in-field (1..14). The number of the module within the field, with 1 at the center of the PFI.
    cm: Cobra-in-module (1..57). 1 is the bottom-left cobra in a module when looked at with the wide (29-cobra) board down. Increasing as you move across the module.
    mod: Cobra module. There are 84 of these, numbered 1 through 42 with A and B suffixes.
    x,y: Position of the center of the cobra on the focal plane.
    r: Radius of the center of the cobra.
    sp: Spectrograph that the cobra feeds (1..4)
    fh: Fiber hole (1..651). This is the position in the spectrograph slit head.
    sfib: Science fiber (1..2394). This is a unique identifier for each science fiber.
    id: Identifier of the USCONNEC connector hole at the Cable B-C interface.
    fiberId: The fiber identifier (1..2604). This is a unique identifier for each fiber (both science and engineering). fiberId = 651*(sp-1)+fh.

For more details, see cobras.pdf
