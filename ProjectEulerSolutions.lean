import «1»

def p1solve := ProjectEulerSolutions.P1.solve

def p1equiv : ProjectEulerStatements.P1.naive = p1solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P1.equiv
