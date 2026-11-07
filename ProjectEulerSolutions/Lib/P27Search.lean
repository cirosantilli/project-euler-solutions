import ProjectEulerStatements.Lib.QuadraticPrimes
import ProjectEulerStatements.Lib.QuadraticPrimeWinner
import Mathlib.Tactic.NormNum

/-! Pruned quadratic-prime candidate traversal and its unique maximizing pair.
The same finite search model is used by the Python and standalone Lean proofs. -/
namespace ProjectEulerSolutions.P27Search
open ProjectEulerStatements.P27
set_option Elab.async false

structure Best where
  length : Int
  a : Int
  b : Int
  deriving DecidableEq

def admissible (a b : Int) : Prop :=
  (if b ≠ 2 then a % 2 ≠ 0 else a % 2 = 0) ∧ isPrimeInt (1 + a + b) = true
instance (a b : Int) : Decidable (admissible a b) := by unfold admissible; infer_instance

def step (a b : Int) (best : Best) : Best :=
  if admissible a b ∧ best.length < (consecPrimeLen a b : Int)
    then ⟨consecPrimeLen a b, a, b⟩ else best

def inner (b start : Int) : Nat → Best → Best
  | 0, best => best
  | count + 1, best => inner b (start + 1) count (step start b best)

theorem inner_eq_foldl (b start : Int) (count : Nat) (best : Best) :
    inner b start count best =
      (List.range count).foldl (fun best (i : Nat) => step (start + i) b best) best := by
  induction count generalizing start best with
  | zero => rfl
  | succ count ih =>
      rw [inner, ih, List.range_succ_eq_map, List.foldl_cons]
      simp only [Nat.cast_zero, add_zero, List.foldl_map]
      apply congrArg (fun f : Best → Nat → Best => (List.range count).foldl f (step start b best))
      funext previous index
      congr 1
      push_cast
      omega

def search (values : List Int) : Best :=
  values.foldl (fun best b => inner b (-999) 1999 best) ⟨-1, 0, 0⟩

/-- Candidate order made explicit for connecting the source fold to a mathematical winner. -/
def pairs (candidates : List Int) : List (Int × Int) :=
  candidates.flatMap fun b => (List.range 1999).map fun (i : Nat) => ((-999 : Int) + i, b)

theorem search_eq_fold (candidates : List Int) :
    search candidates = (pairs candidates).foldl (fun best pair => step pair.1 pair.2 best) ⟨-1, 0, 0⟩ := by
  simp only [search, pairs, List.foldl_flatMap, List.foldl_map, inner_eq_foldl]

/-- An already selected winner or a strictly shorter candidate. -/
def Below (target best : Best) : Prop := best = target ∨ best.length < target.length

/-- Every admitted candidate is shorter than the target, except the target itself. -/
def Dominates (target : Best) (pair : Int × Int) : Prop :=
  admissible pair.1 pair.2 →
    (consecPrimeLen pair.1 pair.2 : Int) ≤ target.length ∧
    ((consecPrimeLen pair.1 pair.2 : Int) = target.length → pair = (target.a, target.b))

private theorem step_below (target best : Best) (a b : Int) (before : Below target best)
    (dominates : Dominates target (a, b)) : Below target (step a b best) := by
  by_cases improve : admissible a b ∧ best.length < (consecPrimeLen a b : Int)
  · rw [step, if_pos improve]
    have bounds := dominates improve.1
    dsimp at bounds
    by_cases full : (consecPrimeLen a b : Int) = target.length
    · left
      have same := bounds.2 full
      cases target
      simp_all
    · right
      change (consecPrimeLen a b : Int) < target.length
      omega
  · simpa only [step, if_neg improve] using before

private theorem step_stable (target : Best) (a b : Int) (dominates : Dominates target (a, b)) :
    step a b target = target := by
  have cannot : ¬ (admissible a b ∧ target.length < (consecPrimeLen a b : Int)) := by
    intro improve
    have bound := (dominates improve.1).1
    change (consecPrimeLen a b : Int) ≤ target.length at bound
    omega
  simp only [step, if_neg cannot]

private theorem step_winner (target best : Best) (before : Below target best)
    (admitted : admissible target.a target.b)
    (score : target.length = (consecPrimeLen target.a target.b : Int)) :
    step target.a target.b best = target := by
  rcases before with rfl | shorter
  · simp [step, ← score]
  · simp only [step, admitted, true_and, ← score, if_pos shorter]

private theorem fold_stable (target : Best) (values : List (Int × Int))
    (dominates : ∀ pair ∈ values, Dominates target pair) :
    values.foldl (fun best pair => step pair.1 pair.2 best) target = target := by
  induction values with
  | nil => rfl
  | cons pair rest ih =>
      rw [List.foldl_cons, step_stable target pair.1 pair.2 (dominates pair (by simp))]
      exact ih (fun pair member => dominates pair (by simp [member]))

/-- Strict updates find a unique maximum independently of the candidate list's order. -/
theorem fold_winner (target best : Best) (values : List (Int × Int))
    (before : Below target best) (admitted : admissible target.a target.b)
    (score : target.length = (consecPrimeLen target.a target.b : Int))
    (dominates : ∀ pair ∈ values, Dominates target pair)
    (present : (target.a, target.b) ∈ values) :
    values.foldl (fun best pair => step pair.1 pair.2 best) best = target := by
  induction values generalizing best with
  | nil => simp at present
  | cons pair rest ih =>
      rcases List.mem_cons.mp present with same | later
      · subst pair
        rw [List.foldl_cons, step_winner target best before admitted score]
        exact fold_stable target rest (fun pair member => dominates pair (by simp [member]))
      · rw [List.foldl_cons]
        exact ih (step pair.1 pair.2 best)
          (step_below target best pair.1 pair.2 before (dominates pair (by simp)))
          (fun pair member => dominates pair (by simp [member])) later

theorem mem_pairs (candidates : List Int) (a b : Int) :
    (a, b) ∈ pairs candidates ↔ b ∈ candidates ∧ -999 ≤ a ∧ a ≤ 999 := by
  constructor
  · intro member
    obtain ⟨b', present, member⟩ := List.mem_flatMap.mp member
    obtain ⟨i, inside, equal⟩ := List.mem_map.mp member
    have bound := List.mem_range.mp inside
    have value : (-999 : Int) + i = a := congrArg Prod.fst equal
    have same : b' = b := congrArg Prod.snd equal
    exact ⟨same ▸ present, by omega, by omega⟩
  · rintro ⟨present, lower, upper⟩
    apply List.mem_flatMap.mpr
    refine ⟨b, present, List.mem_map.mpr ⟨(a + 999).toNat, List.mem_range.mpr (by omega), ?_⟩⟩
    apply Prod.ext
    · change (-999 : Int) + (a + 999).toNat = a
      omega
    · rfl

/-- A unique mathematical winner determines the result of the optimized source search. -/
theorem search_winner (candidates : List Int) (target : Best)
    (present : target.b ∈ candidates) (lower : -999 ≤ target.a) (upper : target.a ≤ 999)
    (positive : -1 < target.length) (admitted : admissible target.a target.b)
    (score : target.length = (consecPrimeLen target.a target.b : Int))
    (dominates : ∀ a b, b ∈ candidates → -999 ≤ a → a ≤ 999 → Dominates target (a, b)) :
    search candidates = target := by
  rw [search_eq_fold]
  apply fold_winner target ⟨-1, 0, 0⟩ (pairs candidates) (Or.inr positive) admitted score
  · rintro ⟨a, b⟩ member
    obtain ⟨present, lower, upper⟩ := (mem_pairs candidates a b).mp member
    exact dominates a b present lower upper
  · exact (mem_pairs candidates target.a target.b).mpr ⟨present, lower, upper⟩

private def coefficients (bound : Nat) : List Int :=
  (List.range (2 * bound + 1)).map (fun (i : Nat) => (i : Int) - bound)
private def rectangle (limitA limitB : Nat) : List (Int × Int) :=
  (coefficients limitA).flatMap (fun a => (coefficients limitB).map (a, ·))
private def selectedProduct (best : Option (Int × Int)) : Int :=
  let chosen := best.getD (0, 0)
  chosen.1 * chosen.2
private theorem naive_eq_argmax (limitA limitB : Nat) :
    naive limitA limitB = selectedProduct
      (List.argmax (fun p : Int × Int => consecPrimeLen p.1 p.2) (rectangle limitA limitB)) := by
  simp only [naive, rectangle, coefficients, selectedProduct, bind_pure_comp,
    List.map_eq_map, List.map_map, Function.comp_def, List.foldr_append_eq_append, List.append_nil,
    List.flatMap, Option.getD]
  generalize List.argmax (fun p : Int × Int => consecPrimeLen p.1 p.2) _ = best
  cases best <;> rfl
private theorem mem_coefficients (bound : Nat) (a : Int) :
    a ∈ coefficients bound ↔ -(bound : Int) ≤ a ∧ a ≤ bound := by
  constructor
  · intro member
    obtain ⟨i, inside, equal⟩ := List.mem_map.mp member
    have boundI := List.mem_range.mp inside
    omega
  · rintro ⟨lower, upper⟩
    apply List.mem_map.mpr
    refine ⟨(a + bound).toNat, List.mem_range.mpr (by omega), ?_⟩
    omega
private theorem mem_rectangle (a b : Int) :
    (a, b) ∈ rectangle 999 1000 ↔ -999 ≤ a ∧ a ≤ 999 ∧ -1000 ≤ b ∧ b ≤ 1000 := by
  simp only [rectangle, List.mem_flatMap, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨x, xMem, y, yMem, hx, hy⟩
    subst x
    subst y
    exact ⟨(mem_coefficients 999 a).mp xMem |>.1,
      (mem_coefficients 999 a).mp xMem |>.2,
      (mem_coefficients 1000 b).mp yMem |>.1,
      (mem_coefficients 1000 b).mp yMem |>.2⟩
  · rintro ⟨al, au, bl, bu⟩
    exact ⟨a, (mem_coefficients 999 a).mpr ⟨al,au⟩,
      b, (mem_coefficients 1000 b).mpr ⟨bl,bu⟩, rfl, rfl⟩
theorem reference_winner
    (score : consecPrimeLen (-61) 971 = 71)
    (unique : ∀ a b : Int, a ≤ 999 → b ≤ 1000 →
      71 ≤ consecPrimeLen a b → a = -61 ∧ b = 971)
    (bound : ∀ a b : Int, a ≤ 999 → b ≤ 1000 → consecPrimeLen a b ≤ 71) :
    naive 999 1000 = -59231 := by
  have selected : List.argmax (fun p : Int × Int => consecPrimeLen p.1 p.2) (rectangle 999 1000) = some (-61,971) := by
    apply List.argmax_eq_some_iff.mpr
    refine ⟨(mem_rectangle (-61) 971).mpr (by norm_num), ?_, ?_⟩
    · rintro ⟨a,b⟩ member
      have bounds := (mem_rectangle a b).mp member
      simpa only [score] using bound a b bounds.2.1 bounds.2.2.2
    · rintro ⟨a,b⟩ member atLeast
      have bounds := (mem_rectangle a b).mp member
      obtain ⟨rfl,rfl⟩ := unique a b bounds.2.1 bounds.2.2.2 (by simpa [score] using atLeast)
      exact Nat.le_refl _
  rw [naive_eq_argmax, selected]
  rfl

def candidateValues : List Int :=
  ((List.range' 2 999).filter Nat.Prime).map Int.ofNat

private theorem mem_primes (p : Nat) :
    p ∈ (List.range' 2 999).filter Nat.Prime ↔ Nat.Prime p ∧ p ≤ 1000 := by
  simp only [List.mem_filter, List.mem_range', decide_eq_true_eq]
  constructor
  · rintro ⟨⟨i, bound, rfl⟩, prime⟩; exact ⟨prime, by omega⟩
  · rintro ⟨prime, upper⟩
    exact ⟨⟨p-2, by have := prime.two_le; omega, by have := prime.two_le; omega⟩, prime⟩

private def target : Best := ⟨71, -61, 971⟩

theorem optimized_winner : search candidateValues = target := by
  apply search_winner candidateValues target
  · apply List.mem_map.mpr
    exact ⟨971, (mem_primes 971).mpr ⟨by norm_num, by omega⟩, rfl⟩
  · decide
  · decide
  · decide
  · norm_num [target, admissible, isPrimeInt, Int.toNat]
  · simp [target, ProjectEulerStatements.QuadraticPrimeWinner.canonical_length]
  · intro a b member _ upper admitted
    obtain ⟨p, present, rfl⟩ := List.mem_map.mp member
    have upperP := ((mem_primes p).mp present).2
    have upperB : (p : Int) ≤ 1000 := by exact_mod_cast upperP
    constructor
    · have bound := ProjectEulerStatements.QuadraticPrimeWinner.score_le a p upper upperB
      change (consecPrimeLen a p : Int) ≤ 71
      exact_mod_cast bound
    · intro same
      have long : 71 ≤ consecPrimeLen a p := by change (consecPrimeLen a p : Int) = 71 at same; omega
      obtain ⟨aValue, bValue⟩ := ProjectEulerStatements.QuadraticPrimeWinner.unique_long a p upper upperB long
      exact Prod.ext aValue bValue

def optimizedAnswer : Int := (search candidateValues).a * (search candidateValues).b

theorem optimized_answer : optimizedAnswer = -59231 := by
  rw [optimizedAnswer, optimized_winner]
  rfl

theorem reference_answer : ProjectEulerStatements.P27.naive 999 1000 = -59231 :=
  reference_winner ProjectEulerStatements.QuadraticPrimeWinner.canonical_length
    ProjectEulerStatements.QuadraticPrimeWinner.unique_long
    ProjectEulerStatements.QuadraticPrimeWinner.score_le

end ProjectEulerSolutions.P27Search
