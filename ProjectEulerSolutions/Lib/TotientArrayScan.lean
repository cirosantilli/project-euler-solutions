import ProjectEulerSolutions.Lib.TotientArraySieveCorrect
import ProjectEulerSolutions.Lib.DigitHistogram
import ProjectEulerStatements.Lib.TotientPermutationScan

set_option linter.style.header false

/-! Array-backed ratio selection with moving decimal intervals and unbounded digit counts. -/
namespace ProjectEulerSolutions.TotientArrayScan
open ProjectEulerStatements TotientPermutation
set_option Elab.async false

def digitSignature (n : Nat) : List Nat := DigitHistogram.signature n

structure State where
  best : Nat
  bestPhi : Nat
  low : Nat
  high : Nat
  deriving DecidableEq, Repr

def boundary (n : Nat) (state : State) : State :=
  if n = state.high then { state with low := state.high, high := state.high * 10 } else state

/-- All candidate arithmetic reads the computed array, including the cached incumbent ratio. -/
def evaluate (n pn : Nat) (state : State) : State :=
  if pn < state.low ∨ state.high ≤ pn then state
  else if ((n : Int) - pn) % 9 ≠ 0 then state
  else if digitSignature n ≠ digitSignature pn then state
  else if state.best = 0 ∨ n * state.bestPhi < state.best * pn then
    { state with best := n, bestPhi := pn }
  else state

def step (phi : Array Nat) (n : Nat) (state : State) : State :=
  evaluate n phi[n]! (boundary n state)

def scan (limit n : Nat) (phi : Array Nat) (state : State) : Nat :=
  if n ≤ limit then scan limit (n + 1) phi (step phi n state) else state.best
termination_by limit + 1 - n
decreasing_by omega

def initial : State := ⟨0, 1, 1, 10⟩

def solveCore (limit : Nat) : Nat :=
  scan limit 2 (TotientArraySieve.computePhiLinear limit) initial

def solve (limit : Int) : Nat := if limit < 2 then 0 else solveCore limit.toNat

end ProjectEulerSolutions.TotientArrayScan
