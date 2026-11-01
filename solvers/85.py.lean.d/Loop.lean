import solvers.«85.py.lean.d».Candidates

/-! The optimized Python outer loop realizes the complete rectangle search. -/
namespace solvers.«85».Optimized
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangleApproximation ProjectEulerSolutions.RectangleSearch
open ProjectEulerSolutions.TriPentHex (triangular)
set_option Elab.async false

abbrev verticalExpression : Expr := .binOp
  (.binOp (.name "m") .mult (.binOp (.name "m") .add (.constant (.int 1))))
  .floorDiv (.constant (.int 2))
abbrev stopCondition : Expr := .compare (.binOp (.name "vertical") .mult (.name "vertical"))
  [(.gt, .binOp (.name "target") .add (.name "best_diff"))]
abbrev returnExpression : Expr := .tuple
  [.name "best_m", .name "best_n", .binOp (.name "best_m") .mult (.name "best_n")]
abbrev quotientExpression : Expr := .builtinCall .max
  [.constant (.int 0), .binOp (.name "target") .floorDiv (.name "vertical")]
abbrev belowExpression : Expr := .builtinCall .max [.name "m",
  .binOp (.binOp (.call (.name "isqrt") [.binOp (.constant (.int 1)) .add
    (.binOp (.constant (.int 8)) .mult (.name "quotient"))]) .sub (.constant (.int 1)))
    .floorDiv (.constant (.int 2))]
abbrev advanceCode : List Stmt :=
  [.assign [.name "quotient"] quotientExpression, .assign [.name "below"] belowExpression,
   candidateLoop, .augAssign (.name "m") .add (.constant (.int 1))]
abbrev outerBody : List Stmt :=
  [.assign [.name "vertical"] verticalExpression,
   .ifThen stopCondition [.return returnExpression] []] ++ advanceCode
abbrev outerLoop : Stmt := .whileDo (.constant (.bool true)) outerBody []
abbrev resultItems (grid : Grid) : List Mutable.Value :=
  [number grid.1, number grid.2, number (grid.1 * grid.2)]

theorem vertical_evaluate (m : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getM : env "m" = some (number m)) :
    Evaluates program env verticalExpression heap (number (triangular m)) heap globals := by
  apply Evaluates.binary (middle := heap) (afterRight := heap)
    (av := number (m * (m + 1))) (bv := number 2)
  · exact Evaluates.binary (av := number m) (bv := number (m + 1))
      (middle := heap) (afterRight := heap) (Evaluates.lookup getM)
      (Evaluates.binary (value := number (m + 1)) (finalHeap := heap)
        (Evaluates.lookup getM) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number]))
      (by simp [Mutable.binary, scalarBinary, number])
  · exact Evaluates.constant
  · simp [Mutable.binary, scalarBinary, number, triangular, Int.natCast_ediv]

theorem stop_evaluate (target : Int) (m : Nat) (best : Grid) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (ready : Locals env target m best)
    (getVertical : env "vertical" = some (number (triangular m))) :
    Evaluates program env stopCondition heap
      (.scalar (.bool (decide (target + error target best < (count m m : Int))))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number (count m m)) (bv := integer (target + error target best))
  · exact Evaluates.binary (Evaluates.lookup getVertical) (Evaluates.lookup getVertical)
      (by simp [Mutable.binary, scalarBinary, number, count])
  · exact Evaluates.binary (Evaluates.lookup ready.targetValue) (Evaluates.lookup ready.differenceValue) rfl
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem return_evaluate (target : Int) (m : Nat) (best : Grid) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (ready : Locals env target m best) :
    Evaluates program env returnExpression heap (.ref heap.length)
      (heap ++ [.tuple (resultItems best)]) globals := by
  apply Evaluates.tuple
  · exact .cons (Evaluates.lookup ready.bestM) (.cons (Evaluates.lookup ready.bestN)
      (.cons (Evaluates.binary (Evaluates.lookup ready.bestM) (Evaluates.lookup ready.bestN)
        (by simp [Mutable.binary, scalarBinary, number])) .nil))
  · simp [resultItems]

theorem advance_result (target : Int) (m : Nat) (positive : 0 < m) (best : Grid)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (ready : Locals env target m best) (getVertical : env "vertical" = some (number (triangular m))) :
    ∃ nextEnv nextHeap, BlockContinues program env advanceCode heap nextEnv nextHeap globals ∧
      Locals nextEnv target (m + 1) (fastRow target m best) ∧ heap <+: nextHeap := by
  let quotientEnv := setLocal env "quotient" (number (target.toNat / triangular m))
  let belowEnv := setLocal quotientEnv "below" (number (rowBelow target m))
  have quotientDone : BlockContinues program env [.assign [.name "quotient"] quotientExpression]
      heap quotientEnv heap globals :=
    BlockContinues.assign (quotient_evaluate target positive env heap globals ready.targetValue getVertical)
  have belowDone : BlockContinues program quotientEnv [.assign [.name "below"] belowExpression]
      heap belowEnv heap globals := BlockContinues.assign (below_evaluate target m quotientEnv heap globals
        (by simpa [quotientEnv, setLocal] using ready.rowValue) (by simp [quotientEnv, setLocal]))
  have belowReady : Locals belowEnv target m best := by
    rcases ready with ⟨ht, hm, hb, hn, hd⟩
    constructor <;> simp_all [belowEnv, quotientEnv, setLocal]
  obtain ⟨rowEnv, rowHeap, rowDone, rowReady, preserved⟩ :=
    candidates_result target m best belowEnv heap globals belowReady (by simp [belowEnv, setLocal])
  let nextEnv := setLocal rowEnv "m" (number (m + 1))
  have increment : BlockContinues program rowEnv [.augAssign (.name "m") .add (.constant (.int 1))]
      rowHeap nextEnv rowHeap globals := by
    apply BlockContinues.augAssign rowReady.rowValue (right := number 1) Evaluates.constant
    simp [inplace, Mutable.binary, scalarBinary, number]
  refine ⟨nextEnv, rowHeap, quotientDone.append (belowDone.append (rowDone.append increment)), ?_, preserved⟩
  rcases rowReady with ⟨ht, hm, hb, hn, hd⟩
  constructor <;> simp_all [nextEnv, setLocal]

/-- Every terminating mathematical scan is realized by the original Python while loop. -/
theorem loop_result {target : Int} {m : Nat} {best final : Grid}
    (run : RowsRun target (optimizedRow target) m best final) (positive : 0 < m)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (ready : Locals env target m best) :
    ∃ address finalHeap, BlockReturns program env [outerLoop] heap (.ref address) finalHeap globals ∧
      finalHeap[address]? = some (.tuple (resultItems final)) ∧ heap <+: finalHeap := by
  induction run generalizing env heap with
  | stop m best finished =>
      let verticalEnv := setLocal env "vertical" (number (triangular m))
      have verticalReady : Locals verticalEnv target m best := by
        rcases ready with ⟨ht, hm, hb, hn, hd⟩
        constructor <;> simp_all [verticalEnv, setLocal]
      have tested := stop_evaluate target m best verticalEnv heap globals verticalReady (by simp [verticalEnv, setLocal])
      refine ⟨heap.length, heap ++ [.tuple (resultItems best)], ?_, by simp, List.prefix_append _ _⟩
      apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
      apply BlockReturns.assign (middle := heap) (value := number (triangular m))
        (vertical_evaluate m env heap globals ready.rowValue)
      apply BlockReturns.branch (flag := true) (by simpa only [finished, decide_true] using tested) rfl
      exact BlockReturns.return_value (return_evaluate target m best verticalEnv heap globals verticalReady)
  | step m best next final active searched tail ih =>
      change next = fastRow target m best at searched
      subst next
      let verticalEnv := setLocal env "vertical" (number (triangular m))
      have verticalReady : Locals verticalEnv target m best := by
        rcases ready with ⟨ht, hm, hb, hn, hd⟩
        constructor <;> simp_all [verticalEnv, setLocal]
      have tested := stop_evaluate target m best verticalEnv heap globals verticalReady (by simp [verticalEnv, setLocal])
      have continuing : ¬ target + error target best < (count m m : Int) := by omega
      obtain ⟨nextEnv, nextHeap, advanced, nextReady, preserved⟩ :=
        advance_result target m positive best verticalEnv heap globals verticalReady (by simp [verticalEnv, setLocal])
      obtain ⟨address, finalHeap, continued, found, restPreserved⟩ := ih (by omega) nextEnv nextHeap nextReady
      refine ⟨address, finalHeap, ?_, found, preserved.trans restPreserved⟩
      apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
      apply BlockReturns.assign (middle := heap) (value := number (triangular m))
        (vertical_evaluate m env heap globals ready.rowValue)
      apply BlockReturns.branch (flag := false) (by simpa only [continuing, decide_false] using tested) rfl
      change BlockReturns program verticalEnv (advanceCode ++ [outerLoop]) heap (.ref address) finalHeap globals
      exact advanced.returns continued

end solvers.«85».Optimized
