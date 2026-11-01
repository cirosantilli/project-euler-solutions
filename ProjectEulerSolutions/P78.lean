import ProjectEulerStatements.P78
import ProjectEulerStatements.Lib.PartitionPentagonalLoop

/-! An executable residue-table search with a proved partial denotation and no search cutoff. -/
namespace ProjectEulerSolutions.P78
open ProjectEulerStatements.PartitionPentagonalLoop ProjectEulerStatements.PartitionRecurrence
open ProjectEulerStatements.UnboundedSearch
set_option Elab.async false

def sumFrom (p : Array Int) (n k : Nat) (total : Int) : Int :=
  if n < first k then total
  else
    let total := total + sign k * p.getD (n-first k) 0
    let total := if second k ≤ n then total + sign k * p.getD (n-second k) 0 else total
    sumFrom p n (k+1) total
termination_by n+1-k
decreasing_by
  have := first_ge_index k
  omega

theorem sumFrom_eq_tail (p : Array Int) (n k : Nat) (total : Int) (positive : 0 < k) :
    sumFrom p n k total = total + tail (fun j => p.getD j 0) n k := by
  rw [sumFrom]
  split
  · rename_i outside
    rw [tail_done _ n k positive outside, add_zero]
  · rename_i inside
    have bounded : k ≤ n := (first_ge_index k).trans (by omega)
    rw [sumFrom_eq_tail p n (k+1) _ (by omega), tail_step _ n k bounded]
    by_cases extra : second k ≤ n <;>
      simp [piece, show first k ≤ n by omega, extra] <;> ring
termination_by n+1-k
decreasing_by
  have := first_ge_index k
  omega

def nextPartition (p : Array Int) (n : Nat) : Int := sumFrom p n 1 0

def fill (n i : Nat) (p : Array Int) : Array Int :=
  if n < i then p else fill n (i+1) (p.set! i (nextPartition p i))
termination_by n+1-i
decreasing_by omega

def partitionsUpto (n : Nat) : Array Int := fill n 1 ((Array.replicate (n+1) 0).set! 0 1)

def Valid (modulus : Int) (n : Nat) (p : Array Int) : Prop :=
  p.size = n ∧ ∀ j < n, p.getD j 0 % modulus = count j % modulus

theorem valid_initial (modulus : Int) : Valid modulus 1 #[1] := by
  refine ⟨rfl, ?_⟩
  intro j inside
  have zero : j = 0 := by omega
  subst j
  simp [Array.getD, count]

theorem nextPartition_mod (modulus : Int) (n : Nat) (p : Array Int)
    (positive : 0 < n) (valid : Valid modulus n p) :
    nextPartition p n % modulus = count n % modulus := by
  rw [nextPartition, sumFrom_eq_tail p n 1 0 (by omega), zero_add,
    tail_start_mod _ n modulus valid.2 positive]

theorem valid_push (modulus : Int) (n : Nat) (p : Array Int) (valid : Valid modulus n p) :
    Valid modulus (n+1) (p.push (count n % modulus)) := by
  have size := valid.1
  refine ⟨by simp [size], ?_⟩
  intro j inside
  by_cases earlier : j < n
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_push, if_neg (by omega : j ≠ p.size)]
    simpa only [Array.getD_eq_getD_getElem?] using valid.2 j earlier
  · have same : j = n := by omega
    subst j
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_push, if_pos (by omega : n = p.size)]
    simp

abbrev State := Nat × Array Int

def step (modulus : Int) (state : State) : Sum Nat State :=
  let n := state.1
  let residue := nextPartition state.2 n % modulus
  if residue = 0 then .inl n else .inr (n+1, state.2.push residue)

def solve (modulus : Int) : Option Nat := run (step modulus) (1, #[1])

def Related (modulus : Int) (state : State) (n : Nat) : Prop :=
  state.1 = n ∧ 0 < n ∧ Valid modulus n state.2

theorem step_simulation (modulus : Int) (state : State) (n : Nat) (related : Related modulus state n) :
    (∃ result, step modulus state = .inl result ∧ ProjectEulerStatements.P78.step modulus n = .inl result) ∨
    (∃ nextState nextN, step modulus state = .inr nextState ∧
      ProjectEulerStatements.P78.step modulus n = .inr nextN ∧ Related modulus nextState nextN) := by
  rcases state with ⟨i, p⟩
  obtain ⟨rfl, positive, valid⟩ := related
  have residue := nextPartition_mod modulus i p positive valid
  by_cases divides : count i % modulus = 0
  · left
    refine ⟨i, ?_, ?_⟩
    · simp [step, residue, divides]
    · change (if count i % modulus = 0 then Sum.inl i else Sum.inr (i+1)) = _
      rw [if_pos divides]
  · right
    refine ⟨(i+1, p.push (count i % modulus)), i+1, ?_, ?_, rfl, by omega,
      valid_push modulus i p valid⟩
    · simp [step, residue, divides]
    · change (if count i % modulus = 0 then Sum.inl i else Sum.inr (i+1)) = _
      rw [if_neg divides]

theorem solve_result (modulus : Int) : solve modulus = ProjectEulerStatements.P78.naive modulus :=
  run_equiv (step modulus) (ProjectEulerStatements.P78.step modulus) (Related modulus)
    (step_simulation modulus) (1, #[1]) 1 ⟨rfl, by omega, valid_initial modulus⟩

end ProjectEulerSolutions.P78
