import ProjectEulerStatements.P74
import ProjectEulerSolutions.Lib.DigitFactorialWalk

/-! Hash-map memoization stores complete cycles and unwinds each uncached tail. -/
namespace ProjectEulerSolutions.P74
open ProjectEulerStatements.DigitFactorialChains
open DigitFactorialCache
set_option Elab.async false

def chainLength (start : Nat) (entries : Cache) : Nat × Cache :=
  let walked := DigitFactorialWalk.fromStart start entries
  let stored := DigitFactorialWalk.finalCache entries start [] walked
  ((lookup stored start).getD 0, stored)

theorem chainLength_correct (start : Nat) (entries : Cache) (valid : Valid entries) :
    (chainLength start entries).1 = chainLen start ∧ Valid (chainLength start entries).2 := by
  obtain ⟨storedValid, selected⟩ := DigitFactorialWalk.fromStart_correct start entries valid
  exact ⟨by simp only [chainLength, selected, Option.getD_some], storedValid⟩

def picked (target : Int) (first count : Nat) : Nat :=
  ((List.range' first count).filter (fun n => decide ((chainLen n : Int) = target))).length

def scan : Nat → Nat → Nat → Int → Cache → Nat
  | 0, _, total, _, _ => total
  | count + 1, first, total, target, entries =>
      let (length, stored) := chainLength first entries
      scan count (first + 1) (total + if (length : Int) = target then 1 else 0) target stored

theorem scan_correct (count first total : Nat) (target : Int)
    (entries : Cache) (valid : Valid entries) :
    scan count first total target entries = total + picked target first count := by
  induction count generalizing first total entries with
  | zero => simp [scan, picked]
  | succ count ih =>
      have correct := chainLength_correct first entries valid
      simp only [scan]
      rw [ih _ _ _ correct.2, correct.1]
      by_cases chosen : (chainLen first : Int) = target <;>
        simp [picked, List.range'_succ, chosen, Nat.add_assoc, Nat.add_comm]

def solve (limit : Int := 1_000_000) (target : Int := 60) : Int :=
  scan (limit - 1).toNat 1 0 target ∅

theorem solve_result (limit target : Int) : solve limit target =
    ProjectEulerStatements.P74.result limit target := by
  have empty : Valid (∅ : Cache) := DigitFactorialCache.empty_valid
  simp only [solve, scan_correct _ _ _ _ _ empty, Nat.zero_add,
    picked, ProjectEulerStatements.P74.result]

end ProjectEulerSolutions.P74
