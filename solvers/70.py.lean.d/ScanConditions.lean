import solvers.«70.py.lean.d».Sieve
import solvers.«70.py.lean.d».Signature
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerStatements.Lib.TotientPermutationMinimum

/-! Exact table reads, digit-count comparison, and predicates of the optimized scan. -/
namespace solvers.«70».scan
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitHistogram
open ProjectEulerStatements TotientPermutation
set_option Elab.async false

abbrev solverBody : List Stmt := ((findFunctionIn program.body "solve").getD ([], [])).2
abbrev scanBody : List Stmt :=
  match ((executableBody solverBody)[6]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []
abbrev signatureCondition : Expr := .compare (.call (.name "digit_signature") [.name "n"])
  [(.notEq, .call (.name "digit_signature") [.name "pn"])]
abbrev widthCondition : Expr := .boolOp .or
  [.compare (.name "pn") [(.lt, .name "low")], .compare (.name "pn") [(.gtE, .name "high")]]
abbrev residueCondition : Expr :=
  .compare (.binOp (.binOp (.name "n") .sub (.name "pn")) .mod (.constant (.int 9)))
    [(.notEq, .constant (.int 0))]
abbrev improveCondition : Expr := .boolOp .or
  [.compare (.name "best_n") [(.eq, .constant (.int 0))],
    .compare (.binOp (.name "n") .mult (.name "best_phi"))
      [(.lt, .binOp (.name "best_n") .mult (.name "pn"))]]

theorem scan_conditions_source : scanBody.drop 2 =
    [.ifThen widthCondition [.continueLoop] [], .ifThen residueCondition [.continueLoop] [],
      .ifThen signatureCondition [.continueLoop] [],
      .ifThen improveCondition
        [.assign [.name "best_n"] (.name "n"), .assign [.name "best_phi"] (.name "pn")] []] := rfl

def PhiTable (limit address : Nat) (heap : Mutable.Heap) : Prop :=
  heap[address]? = some (.list (sieve.entries (limit + 1) (fun n => number n.totient)))

theorem PhiTable.append {limit address : Nat} {heap : Mutable.Heap}
    (found : PhiTable limit address heap) (extra : Mutable.Heap) :
    PhiTable limit address (heap ++ extra) :=
  ProjectEulerSolutions.HeapCollection.prefix_found heap extra address _ found

theorem read_phi_result (limit address n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : PhiTable limit address heap) (within : n ≤ limit)
    (getPhi : env "phi" = some (.ref address)) (getN : env "n" = some (number n)) :
    Evaluates program env (.subscript (.name "phi") (.name "n")) heap (number n.totient) heap := by
  apply Evaluates.subscript (by intro a b c impossible; cases impossible)
    (Evaluates.lookup getPhi) (Evaluates.lookup getN)
  simpa [sieve.entries] using Mutable.getItem_list heap address n
    (sieve.entries (limit + 1) (fun n => number n.totient)) found
    (by simp [sieve.entries]; omega)

def signatureObjects (n : Nat) : Mutable.Heap :=
  [.list [number 0], .list (numbers (signature n))]

/-- Counting and sorting recognize exactly the same decimal digit permutations. -/
theorem histogram_permutation (a b : Nat) :
    decide (signature a = signature b) = isPermutation a b := by
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq, signature_eq_iff, permutation_iff]

private theorem compare_counts (heap : Mutable.Heap) (a b : Nat) (left right : List Nat)
    (distinct : a ≠ b) (leftFound : heap[a]? = some (.list (numbers left)))
    (rightFound : heap[b]? = some (.list (numbers right))) :
    HeapCalls.compare .notEq (.ref a) (.ref b) heap =
      some (.ok (flag (!decide (left = right))), heap) := by
  have equality : equalLists (numbers left) (numbers right) = some (decide (left = right)) := by
    simpa only [numbers, List.map_map, Function.comp_def,
      List.map_inj_right (fun (x y : Nat) same => Int.natCast_inj.mp same)] using
      equalLists_ints (left.map (fun (n : Nat) => (n : Int)))
        (right.map (fun (n : Nat) => (n : Int)))
  simp [HeapCalls.compare, Mutable.array, leftFound, rightFound, distinct, equality,
    show (CmpOp.notEq == CmpOp.eq) = false from rfl]

/-- Both count arrays are compared by contents after their fresh allocations. -/
theorem signature_condition_result (n pn : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getPN : env "pn" = some (number pn)) :
    Evaluates program env signatureCondition heap (flag (!(isPermutation n pn)))
      (heap ++ signatureObjects n ++ signatureObjects pn) := by
  have left : ReturnsValue program "digit_signature" [number n] heap
      (.ref (heap.length + 1)) (heap ++ signatureObjects n) := by
    simpa only [signatureObjects, Int.toNat_natCast] using digit_signature.result n heap
  have right : ReturnsValue program "digit_signature" [number pn] (heap ++ signatureObjects n)
      (.ref ((heap ++ signatureObjects n).length + 1))
      (heap ++ signatureObjects n ++ signatureObjects pn) := by
    simpa only [signatureObjects, Int.toNat_natCast] using
      digit_signature.result pn (heap ++ signatureObjects n)
  apply Evaluates.compare (av := .ref (heap.length + 1))
    (bv := .ref ((heap ++ signatureObjects n).length + 1))
    (middle := heap ++ signatureObjects n)
    (afterRight := heap ++ signatureObjects n ++ signatureObjects pn)
  · exact Evaluates.call (.cons (Evaluates.lookup getN) .nil) left
  · exact Evaluates.call (.cons (Evaluates.lookup getPN) .nil) right
  · rw [← histogram_permutation]
    apply compare_counts
    · simp [signatureObjects]
    · simp [signatureObjects, List.append_assoc]
    · simp [signatureObjects, List.append_assoc, Nat.add_assoc]

theorem residue_condition_result (n pn : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getPN : env "pn" = some (number pn)) :
    Evaluates program env residueCondition heap
      (flag (decide (((n : Int) - pn) % 9 ≠ 0))) heap := by
  apply Evaluates.compare (av := number (((n : Int) - pn) % 9)) (bv := number 0)
    (middle := heap) (afterRight := heap)
  · apply Evaluates.binary (av := number ((n : Int) - pn)) (bv := number 9)
      (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup getN) (Evaluates.lookup getPN)
        (by simp [Mutable.binary, scalarBinary])
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem width_condition_result (pn low high : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getPN : env "pn" = some (number pn)) (getLow : env "low" = some (number low))
    (getHigh : env "high" = some (number high)) :
    Evaluates program env widthCondition heap (flag (decide (pn < low ∨ high ≤ pn))) heap := by
  have lower : Evaluates program env (.compare (.name "pn") [(.lt, .name "low")]) heap
      (flag (decide (pn < low))) heap := by
    exact Evaluates.compare (middle := heap) (afterRight := heap)
      (Evaluates.lookup getPN) (Evaluates.lookup getLow)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  have upper : Evaluates program env (.compare (.name "pn") [(.gtE, .name "high")]) heap
      (flag (decide (high ≤ pn))) heap := by
    exact Evaluates.compare (middle := heap) (afterRight := heap)
      (Evaluates.lookup getPN) (Evaluates.lookup getHigh)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases small : pn < low
  · simpa only [small, true_or, decide_true] using
      (Evaluates.boolean_stop (op := .or)
        (rest := [.compare (.name "pn") [(.gtE, .name "high")]])
        (by simpa only [small, decide_true] using lower) rfl rfl)
  · apply Evaluates.boolean_next
      (by simpa only [small, decide_false] using lower) rfl rfl
    simpa only [small, false_or] using Evaluates.boolean_last upper rfl

theorem improve_condition_result (n pn best bestPhi : Nat) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (getN : env "n" = some (number n))
    (getPN : env "pn" = some (number pn)) (getBest : env "best_n" = some (number best))
    (getBestPhi : env "best_phi" = some (number bestPhi)) :
    Evaluates program env improveCondition heap
      (flag (decide (best = 0 ∨ n * bestPhi < best * pn))) heap := by
  have zeroTest : Evaluates program env
      (.compare (.name "best_n") [(.eq, .constant (.int 0))]) heap
      (flag (decide (best = 0))) heap :=
    Evaluates.compare (middle := heap) (afterRight := heap)
      (Evaluates.lookup getBest) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  have crossTest : Evaluates program env
      (.compare (.binOp (.name "n") .mult (.name "best_phi"))
        [(.lt, .binOp (.name "best_n") .mult (.name "pn"))]) heap
      (flag (decide (n * bestPhi < best * pn))) heap := by
    apply Evaluates.compare (av := number (n * bestPhi : Nat)) (bv := number (best * pn : Nat))
      (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup getN) (Evaluates.lookup getBestPhi)
        (by simp [Mutable.binary, scalarBinary])
    · exact Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup getBest) (Evaluates.lookup getPN)
        (by simp [Mutable.binary, scalarBinary])
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      norm_cast
  by_cases zero : best = 0
  · simpa only [zero, true_or, decide_true] using
      (Evaluates.boolean_stop (op := .or)
        (rest := [.compare (.binOp (.name "n") .mult (.name "best_phi"))
          [(.lt, .binOp (.name "best_n") .mult (.name "pn"))]])
        (by simpa only [zero, decide_true] using zeroTest) rfl rfl)
  · apply Evaluates.boolean_next
      (by simpa only [zero, decide_false] using zeroTest) rfl rfl
    simpa only [zero, false_or] using Evaluates.boolean_last crossTest rfl

end solvers.«70».scan
