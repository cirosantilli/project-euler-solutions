import ProjectEulerStatements.P9
import Mathlib.Tactic
namespace ProjectEulerSolutions.P9

def euclidS (m n : Nat) : Nat :=
  2 * m * (m + n)

def euclidK (total m n : Nat) : Nat :=
  total / euclidS m n

def euclidA (total m n : Nat) : Nat :=
  euclidK total m n * (m * m - n * n)

def euclidB (total m n : Nat) : Nat :=
  euclidK total m n * (2 * m * n)

def euclidC (total m n : Nat) : Nat :=
  euclidK total m n * (m * m + n * n)

def candidateValid (total m n : Nat) : Prop :=
  let a := euclidA total m n
  let b := euclidB total m n
  let c := euclidC total m n
  total % euclidS m n = 0 ∧
    a > 0 ∧ ((a < b ∧ b < c) ∨ (b < a ∧ a < c)) ∧
      a * a + b * b = c * c ∧ a + b + c = total

instance candidateValidDecidable (total m n : Nat) : Decidable (candidateValid total m n) := by
  unfold candidateValid
  infer_instance

def candidate (total m n : Nat) : Nat :=
  if candidateValid total m n then
    euclidA total m n * euclidB total m n * euclidC total m n
  else
    0

def loopN (total m n best : Nat) : Nat :=
  if n >= m then
    best
  else
    loopN total m (n + 1) (Nat.max best (candidate total m n))
termination_by m - n
decreasing_by
  all_goals omega

def loopM (total m best : Nat) : Nat :=
  if hstop : 2 * m * (m + 1) > total then
    best
  else
    loopM total (m + 1) (Nat.max best (loopN total m 1 0))
termination_by total + 1 - m
decreasing_by
  have hm_le : m ≤ total := by
    have hmul : m ≤ 2 * m * (m + 1) := by nlinarith
    exact le_trans hmul (le_of_not_gt hstop)
  omega

def directTripletProducts (total : Nat) : Finset Nat :=
  (((Finset.Icc 1 total).product (Finset.Icc 1 total)).filter (fun p =>
    let a := p.1
    let b := p.2
    let c := total - a - b
    a < b ∧ b < c ∧ a ^ 2 + b ^ 2 = c ^ 2
  )).image (fun p =>
    let a := p.1
    let b := p.2
    let c := total - a - b
    a * b * c)

def directSolve (total : Nat) : Nat :=
  if h : (directTripletProducts total).Nonempty then
    (directTripletProducts total).max' h
  else
    0

def solve (total : Nat) : Nat :=
  Nat.max (loopM total 2 0) (directSolve total)

example : solve 12 = 60 := by
  native_decide
end ProjectEulerSolutions.P9
