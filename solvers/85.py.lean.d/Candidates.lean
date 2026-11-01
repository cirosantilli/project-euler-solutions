import solvers.«85.py.lean.d».Arithmetic
import ProjectEulerSolutions.Lib.RectangleSearch

/-! The optimized row's two candidate updates and their temporary tuples. -/
namespace solvers.«85».Optimized
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangleApproximation ProjectEulerSolutions.RectangleSearch
set_option Elab.async false

abbrev integer (value : Int) : Mutable.Value := .scalar (.int value)

structure Locals (env : HeapCalls.Env) (target : Int) (m : Nat) (best : Grid) : Prop where
  targetValue : env "target" = some (integer target)
  rowValue : env "m" = some (number m)
  bestM : env "best_m" = some (number best.1)
  bestN : env "best_n" = some (number best.2)
  differenceValue : env "best_diff" = some (integer (error target best))

abbrev differenceExpression : Expr := .builtinCall .abs
  [.binOp (.call (.name "rectangle_count") [.name "m", .name "n"]) .sub (.name "target")]
abbrev updateBody : List Stmt :=
  [.assign [.name "best_diff"] (.name "diff"),
   .assign [.tuple [.name "best_m", .name "best_n"]] (.tuple [.name "m", .name "n"])]
abbrev candidateBody : List Stmt :=
  [.assign [.name "diff"] differenceExpression,
   .ifThen (.compare (.name "diff") [(.lt, .name "best_diff")]) updateBody []]
abbrev candidatesExpression : Expr := .tuple
  [.name "below", .binOp (.name "below") .add (.constant (.int 1))]
abbrev candidateLoop : Stmt := .forIn (.name "n") candidatesExpression candidateBody []

theorem difference_evaluate (target : Int) (m n : Nat) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getM : env "m" = some (number m)) (getN : env "n" = some (number n))
    (getTarget : env "target" = some (integer target)) :
    Evaluates program env differenceExpression heap (integer (distance target m n)) heap globals := by
  apply Evaluates.builtin (middle := heap) (values := [integer ((count m n : Int) - target)])
  · exact .cons (Evaluates.binary
      (Evaluates.call (.cons (Evaluates.lookup getM) (.cons (Evaluates.lookup getN) .nil))
        (rectangle_count.result m n heap globals)) (Evaluates.lookup getTarget) rfl) .nil
  · change (some (.ok (integer (((count m n : Int) - target).natAbs : Int)), heap) :
      Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
    simp [integer, distance]

theorem candidate_result (target : Int) (m n : Nat) (best : Grid) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (ready : Locals env target m best) (getN : env "n" = some (number n)) :
    ∃ nextEnv nextHeap, BlockContinues program env candidateBody heap nextEnv nextHeap globals ∧
      Locals nextEnv target m (choose target best (m, n)) ∧ heap <+: nextHeap := by
  let assigned := setLocal env "diff" (integer (distance target m n))
  have assignment : BlockContinues program env [.assign [.name "diff"] differenceExpression]
      heap assigned heap globals :=
    BlockContinues.assign (difference_evaluate target m n env heap globals ready.rowValue getN ready.targetValue)
  have tested : Evaluates program assigned (.compare (.name "diff") [(.lt, .name "best_diff")])
      heap (.scalar (.bool (decide (error target (m, n) < error target best)))) heap globals := by
    apply Evaluates.compare (middle := heap) (afterRight := heap)
      (av := integer (distance target m n)) (bv := integer (error target best))
      (Evaluates.lookup (by simp [assigned, setLocal]))
      (Evaluates.lookup (by simpa [assigned, setLocal] using ready.differenceValue))
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, error]
  by_cases improves : error target (m, n) < error target best
  · let items := [number m, number n]
    let extended := heap ++ [.tuple items]
    let differenceSet := setLocal assigned "best_diff" (integer (distance target m n))
    let updated := assignNames differenceSet ["best_m", "best_n"] items
    have updatedBody : BlockContinues program assigned updateBody heap updated extended globals := by
      apply BlockContinues.append
        (BlockContinues.assign (value := integer (distance target m n)) (Evaluates.lookup (by simp [assigned, setLocal])))
      apply BlockContinues.unpack (names := ["best_m", "best_n"]) rfl
        (value := .ref heap.length) (items := items) (middle := extended)
      · apply Evaluates.tuple (values := items)
        · exact .cons (Evaluates.lookup (by simpa [differenceSet, assigned, setLocal] using ready.rowValue))
            (.cons (Evaluates.lookup (by simpa [differenceSet, assigned, setLocal] using getN)) .nil)
        · simp [items]
      · simp [Mutable.sequence, extended, items]
      · rfl
    have branch : BlockContinues program assigned
        [.ifThen (.compare (.name "diff") [(.lt, .name "best_diff")]) updateBody []]
        heap updated extended globals :=
      BlockContinues.branch (flag := true) (by simpa only [improves, decide_true] using tested) rfl updatedBody
    refine ⟨updated, extended, assignment.append branch, ?_, List.prefix_append _ _⟩
    rw [choose, if_pos improves]
    constructor
    · simpa [updated, differenceSet, assigned, assignNames, setLocal, items] using ready.targetValue
    · simpa [updated, differenceSet, assigned, assignNames, setLocal, items] using ready.rowValue
    · simp [updated, differenceSet, assigned, assignNames, setLocal, items]
    · simp [updated, differenceSet, assigned, assignNames, setLocal, items]
    · simp [updated, differenceSet, assigned, assignNames, setLocal, items, error]
  · refine ⟨assigned, heap, assignment.append ?_, ?_, by simp⟩
    · exact BlockContinues.branch (flag := false)
        (by simpa only [improves, decide_false] using tested) rfl BlockContinues.nil
    · rw [choose, if_neg improves]
      constructor
      · simpa [assigned, setLocal] using ready.targetValue
      · simpa [assigned, setLocal] using ready.rowValue
      · simpa [assigned, setLocal] using ready.bestM
      · simpa [assigned, setLocal] using ready.bestN
      · simpa [assigned, setLocal] using ready.differenceValue

theorem Locals.set_n {env : HeapCalls.Env} {target : Int} {m : Nat} {best : Grid}
    (ready : Locals env target m best) (n : Nat) : Locals (setLocal env "n" (number n)) target m best := by
  rcases ready with ⟨ht, hm, hb, hn, hd⟩
  constructor <;> simp_all [setLocal]

theorem tuple_preserved {heap nextHeap : Mutable.Heap} (preserved : heap <+: nextHeap)
    {address : Nat} {items : List Mutable.Value} (found : heap[address]? = some (.tuple items)) :
    Mutable.sequence (.ref address) nextHeap = some (.ok items, nextHeap) := by
  obtain ⟨inside, value⟩ := List.getElem?_eq_some_iff.mp found
  have retained := List.prefix_iff_getElem?.mp preserved address inside
  rw [value] at retained
  simp [Mutable.sequence, retained]

/-- Execute the actual two-element tuple iteration, including allocations in either update. -/
theorem candidates_result (target : Int) (m : Nat) (best : Grid) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (ready : Locals env target m best)
    (getBelow : env "below" = some (number (rowBelow target m))) :
    ∃ nextEnv nextHeap, BlockContinues program env [candidateLoop] heap nextEnv nextHeap globals ∧
      Locals nextEnv target m (fastRow target m best) ∧ heap <+: nextHeap := by
  let below := rowBelow target m
  let items := [number below, number (below + 1)]
  let extended := heap ++ [.tuple items]
  have found : extended[heap.length]? = some (.tuple items) := by simp [extended]
  have displayed : Evaluates program env candidatesExpression heap (.ref heap.length) extended globals := by
    apply Evaluates.tuple (values := items)
    · exact .cons (Evaluates.lookup getBelow)
        (.cons (Evaluates.binary (Evaluates.lookup getBelow) Evaluates.constant
          (by simp [Mutable.binary, scalarBinary, number, below])) .nil)
    · simp [items]
  obtain ⟨firstEnv, firstHeap, firstDone, firstReady, firstPreserved⟩ :=
    candidate_result target m below best (setLocal env "n" (number below)) extended globals
      (ready.set_n below) (by simp [setLocal])
  obtain ⟨nextEnv, nextHeap, secondDone, nextReady, secondPreserved⟩ :=
    candidate_result target m (below + 1) (choose target best (m, below))
      (setLocal firstEnv "n" (number (below + 1))) firstHeap globals
      (firstReady.set_n _) (by simp [setLocal])
  refine ⟨nextEnv, nextHeap, ?_, nextReady, (List.prefix_append _ _).trans (firstPreserved.trans secondPreserved)⟩
  apply BlockContinues.for_sequence (value := .ref heap.length) (items := items)
    (by intro expressions; simp [candidatesExpression]) displayed (by simp [Mutable.sequence, found])
  apply BlockContinues.sequence_step (source := .ref heap.length) (items := items) (value := number below)
    (by simp [sequenceValue, Mutable.sequence, found]) rfl
  change BlockContinues program _ (candidateBody ++
    [.sequenceStepLoop "n" (.ref heap.length) 1 candidateBody []]) extended nextEnv nextHeap globals
  apply firstDone.append
  apply BlockContinues.sequence_step (source := .ref heap.length) (items := items) (value := number (below + 1))
    (tuple_preserved firstPreserved found) rfl
  change BlockContinues program _ (candidateBody ++
    [.sequenceStepLoop "n" (.ref heap.length) 2 candidateBody []]) firstHeap nextEnv nextHeap globals
  apply secondDone.append
  exact BlockContinues.sequence_done (source := .ref heap.length) (items := items)
    (tuple_preserved (firstPreserved.trans secondPreserved) found) rfl BlockContinues.nil

end solvers.«85».Optimized
