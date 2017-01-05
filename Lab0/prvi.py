#! /usr/bin/env python3

import sympy

# definicija varijabli
x0, x1, x2 = sympy.var('x0,x1,x2')

# definicija funkcija
f = sympy.Matrix([x0 ** 2 + x1 * x2, x0 + x1 + x2])
g = sympy.Matrix([sympy.sin(x0), x1 ** 3 + x0 * x1, x2])

# mapiranje funkcije g na parametre funkcije f
substitution_fg = {'x0': g[0], 'x1': g[1], 'x2': g[2]}

# kompozicija funkcija
fog = f.subs(substitution_fg)

# izravno diferenciranje
Jfog = fog.jacobian([x0, x1, x2])
Jfog.simplify()
sympy.pprint(Jfog)
print()

# Jakobijani pojedinačnih funkcija
Jf = f.jacobian([x0, x1, x2])
Jg = g.jacobian([x0, x1, x2])

# pravilo ulančavanja
Jfog2 = Jf.subs(substitution_fg) * Jg
Jfog2.simplify()
sympy.pprint(Jfog2)
print()

# nastavak prethodnog primjera...
f0 = sympy.Matrix([f[0]])
Jf0 = f0.jacobian([x0, x1, x2])
# Hesseova matrica je Jakobijan gradijenta:
H = Jf0.jacobian([x0, x1, x2])
sympy.pprint(H)
print()
