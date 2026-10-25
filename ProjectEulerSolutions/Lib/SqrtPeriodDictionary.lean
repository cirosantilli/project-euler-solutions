import ProjectEulerStatements.Lib.SqrtPeriods
import lang.py.FormalSW.KeyDictionaryCalls

namespace ProjectEulerSolutions.SqrtPeriodDictionary
open FormalSW.PyAst ProjectEulerStatements.SqrtReducedStates
set_option Elab.async false

def constants (root : Int) (state : State) : List Constant :=
  [.int state.1, .int state.2, .int (coefficient root state)]
def key (root : Int) (state : State) : HashKey := .tuple ((constants root state).map HashKey.scalar)
def stateKey (n index : Nat) : HashKey := key (Nat.sqrt n) (trajectory n index)
def stateValues (n index : Nat) : List Mutable.Value :=
  (constants (Nat.sqrt n) (trajectory n index)).map Mutable.Value.scalar
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

theorem key_equal (root : Int) (left right : State) :
    HashKey.equal (key root left) (key root right) = decide (left = right) := by
  rcases left with ⟨leftM, leftD⟩
  rcases right with ⟨rightM, rightD⟩
  simp [key, constants, HashKey.equal, HashKey.depth, HashKey.listDepth, HashKey.equalFuel,
    List.isEqv, scalarCompare, evalCmpOp, Prod.ext_iff, coefficient]
  by_cases sameM : leftM = rightM <;> by_cases sameD : leftD = rightD <;> simp [sameM, sameD]

theorem stateKey_equal (n left right : Nat) (nonsquare : Nat.sqrt n * Nat.sqrt n ≠ n)
    (leftBound : left ≤ length n) (rightBound : right ≤ length n) :
    HashKey.equal (stateKey n left) (stateKey n right) = decide (left = right) := by
  rw [stateKey, stateKey, key_equal]
  congr 1
  exact propext ⟨trajectory_injective n left right nonsquare leftBound rightBound, by rintro rfl; rfl⟩

theorem stateKey_repeat (n : Nat) (nonsquare : Nat.sqrt n * Nat.sqrt n ≠ n) :
    stateKey n (length n + 1) = stateKey n 1 := by
  simp only [stateKey, trajectory_first_repeat n nonsquare]

/-- Stored indices describe exactly the already visited, distinct trajectory states. -/
def Entries (n count : Nat) (entries : List (HashDict.Entry Mutable.Value)) : Prop :=
  entries.length = count ∧ ∀ index ≤ length n,
    HashDict.lookup entries (stateKey n index) = if index < count then some (number index) else none

theorem entries_empty (n : Nat) : Entries n 0 [] := by
  exact ⟨rfl, by intro index bound; simp⟩

theorem entries_missing (n count : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (view : Entries n count entries) (before : count ≤ length n) :
    HashDict.lookup entries (stateKey n count) = none := by
  simpa using view.2 count before

theorem entries_insert (n count : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (nonsquare : Nat.sqrt n * Nat.sqrt n ≠ n) (view : Entries n count entries)
    (before : count ≤ length n) (original : Mutable.Value) :
    Entries n (count + 1) (HashDict.insert entries (stateKey n count) original (number count)) := by
  rw [HashDict.insert_absent entries _ original _ (entries_missing n count entries view before)]
  refine ⟨by simp [view.1], ?_⟩
  intro index bound
  rw [HashDict.lookup_append, view.2 index bound, HashDict.lookup_cons,
    stateKey_equal n count index nonsquare before bound]
  by_cases earlier : index < count
  · have next : index < count + 1 := by omega
    simp [earlier, next]
  · by_cases same : count = index
    · subst index; simp
    · have outside : ¬ index < count + 1 := by omega
      simp [earlier, same, outside]

theorem entries_repeat (n : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (nonsquare : Nat.sqrt n * Nat.sqrt n ≠ n) (view : Entries n (length n + 1) entries) :
    HashDict.lookup entries (stateKey n (length n + 1)) = some (number 1) := by
  rw [stateKey_repeat n nonsquare, view.2 1 (by have := length_positive n nonsquare; omega)]
  simp [show 1 < length n + 1 by have := length_positive n nonsquare; omega]

end ProjectEulerSolutions.SqrtPeriodDictionary
