import ProjectEulerSolutions.P24
open ProjectEulerSolutions.P24

def main : IO Unit := IO.println (solve ((List.range 10).map Int.ofNat) 1000000)
