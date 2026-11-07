import ProjectEulerSolutions.Lib.IntegerGcdElimination
import ProjectEulerSolutions.Lib.IntegerRowSwap
import ProjectEulerSolutions.Lib.GaussianPivot

/-! Nonsingularity and exact equations throughout gcd-reduced Gaussian elimination. -/
namespace ProjectEulerSolutions.GcdElimination
open scoped BigOperators
set_option Elab.async false

variable {n k : Nat}

abbrev Coefficients (matrix : Matrix (Fin n) (Fin (n + 1)) Int) : Matrix (Fin n) (Fin n) Int :=
  fun row column => matrix row column.castSucc

def Equations (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (value : Fin (n + 1) → ℚ) : Prop :=
  ∀ row, ∑ column, (matrix row column : ℚ) * value column = 0

structure State (original : Matrix (Fin n) (Fin (n + 1)) Int) (processed : Nat) where
  bounded : processed ≤ n
  matrix : Matrix (Fin n) (Fin (n + 1)) Int
  eliminated : ∀ row column, column.val < processed → column.val < row.val → matrix row column = 0
  nonsingular : Function.Injective (Matrix.mulVec (fun row column => (Coefficients matrix row column : ℚ)))
  equations : ∀ value, Equations matrix value ↔ Equations original value

def initial (original : Matrix (Fin n) (Fin (n + 1)) Int)
    (nonsingular : Function.Injective (Matrix.mulVec (fun row column => (Coefficients original row column : ℚ)))) :
    State original 0 where
  bounded := Nat.zero_le n
  matrix := original
  eliminated := by intro row column earlier _; omega
  nonsingular := nonsingular
  equations := fun _ => Iff.rfl

def State.swap {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k)
    (first second : Fin n) (firstBound : k ≤ first.val) (secondBound : k ≤ second.val) : State original k where
  bounded := state.bounded
  matrix := IntegerRowSwap.swap state.matrix first second
  eliminated := IntegerRowSwap.zeros state.matrix first second firstBound secondBound state.eliminated
  nonsingular := IntegerRowSwap.injective (Coefficients state.matrix) first second state.nonsingular
  equations := fun value => (IntegerRowSwap.equations_iff state.matrix first second value).trans (state.equations value)

theorem State.exists_pivot {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k) (unfinished : k < n) :
    ∃ row : Fin n, k ≤ row.val ∧ state.matrix row (⟨k, by omega⟩ : Fin (n + 1)) ≠ 0 := by
  obtain ⟨row, bound, nonzero⟩ := GaussianPivot.exists_pivot_of_injective
    (fun row column => (Coefficients state.matrix row column : ℚ)) state.nonsingular ⟨k, unfinished⟩
    (by
      intro row previous earlier below
      change (state.matrix row previous.castSucc : ℚ) = 0
      exact_mod_cast state.eliminated row previous.castSucc earlier below)
  exact ⟨row, bound, by exact_mod_cast nonzero⟩

/-- A whole column step retains the original rational equations and advances the zero prefix. -/
def State.advance {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k)
    (unfinished : k < n) (nonzero : state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0) : State original (k + 1) where
  bounded := by omega
  matrix := IntegerGcdElimination.reduce state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩
  eliminated := by
    intro row column processed below
    exact IntegerGcdElimination.zeros state.matrix ⟨k, unfinished⟩ state.eliminated row column (by change column.val ≤ k; omega) below
  nonsingular := IntegerGcdElimination.injective state.matrix ⟨k, unfinished⟩ state.nonsingular nonzero
  equations := fun value => (IntegerGcdElimination.equations_iff state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ nonzero value).trans (state.equations value)

def Step {original : Matrix (Fin n) (Fin (n + 1)) Int} (before : State original k) (after : State original (k + 1)) : Prop :=
  ∃ (unfinished : k < n) (candidate : Fin n) (bound : k ≤ candidate.val)
    (nonzero : (before.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound).matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0),
    after = (before.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound).advance unfinished nonzero

theorem State.next {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k) (unfinished : k < n) :
    ∃ after, Step state after := by
  obtain ⟨candidate, bound, nonzero⟩ := state.exists_pivot unfinished
  have swapped : (state.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound).matrix
      ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0 := by
    simpa only [State.swap, IntegerRowSwap.swap, Equiv.swap_apply_left] using nonzero
  exact ⟨(state.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound).advance unfinished swapped,
    unfinished, candidate, bound, swapped, rfl⟩

inductive Runs {original : Matrix (Fin n) (Fin (n + 1)) Int} :
    {first last : Nat} → State original first → State original last → Prop
  | refl {phase : Nat} (state : State original phase) : Runs state state
  | next {first last : Nat} {before : State original first} {middle : State original (first + 1)} {after : State original last} :
      Step before middle → Runs middle after → Runs before after

/-- All n columns can be processed; the argument uses matrix size and nonsingularity, without a cutoff. -/
theorem State.finish {original : Matrix (Fin n) (Fin (n + 1)) Int}
    (count stop : Nat) (state : State original k) (length : k + count = stop) (bound : stop ≤ n) :
    ∃ after : State original stop, Runs state after := by
  induction count generalizing k with
  | zero =>
      have same : k = stop := by omega
      subst k
      exact ⟨state, .refl state⟩
  | succ count ih =>
      obtain ⟨next, step⟩ := state.next (by omega)
      obtain ⟨after, ran⟩ := ih next (by omega)
      exact ⟨after, .next step ran⟩

end ProjectEulerSolutions.GcdElimination
