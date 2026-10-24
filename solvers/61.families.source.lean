import solvers.«61.polygonal.source»

set_option Elab.async false

namespace solvers.«61».generate_4digit_polygonals
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PolygonalFamilies
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev polygonal := ProjectEulerSolutions.PolygonalCycle.polygonal
private abbrev answer := family

private def numbers (values : List Int) : List Mutable.Value := values.map number
private def working (base : Mutable.Heap) (values : List Int) : Mutable.Heap :=
  base ++ [.list (numbers values)]

private def rangeCondition : Expr :=
  .compare (.name "val") [(.gtE, .constant (.int 1000))]
private def filterCode : List Stmt :=
  [.ifThen rangeCondition [.expr (.call (.attribute (.name "nums") "append") [.name "val"])] []]
private def body : List Stmt :=
  [.assign [.name "val"] (.call (.name "polygonal") [.name "s", .name "n"]),
   .ifThen (.compare (.name "val") [(.gtE, .constant (.int 10000))]) [.breakLoop] []] ++
  filterCode ++ [.augAssign (.name "n") .add (.constant (.int 1))]
private def statement : Stmt := .whileDo (.constant (.bool true)) body []
private def code : List Stmt :=
  [.annAssign (.name "nums") (.subscript (.name "list") (.name "int")) (some (.list [])),
   .assign [.name "n"] (.constant (.int 1)), statement, .return (.name "nums")]

theorem code_source :
    executableBody ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).2 = code := rfl

private theorem filter_step (value : Int) (below : value < 10000)
    (env : HeapCalls.Env) (base : Mutable.Heap) (before : List Int)
    (getVal : env "val" = some (number value))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals) :
    BlockContinues program env filterCode (working base before) env
      (working base (before ++ emitted value)) globals := by
  have tested : Evaluates program env rangeCondition (working base before)
      (flag (decide (1000 ≤ value))) (working base before) globals := by
    apply Evaluates.compare (av := number value) (bv := number 1000)
      (middle := working base before) (afterRight := working base before)
      (Evaluates.lookup getVal) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]
  by_cases keep : 1000 ≤ value
  · simp only [emitted, accepted, keep, below, and_self, ↓reduceIte]
    apply BlockContinues.branch (flag := true) (by simpa [keep] using tested) rfl
    have found : (working base before)[base.length]? = some (.list (numbers before)) := by simp [working]
    simpa [working, numbers, List.set_append_right] using
      BlockContinues.discard (Evaluates.append (Evaluates.lookup getNums) found (Evaluates.lookup getVal) found)
  · simp only [emitted, accepted, keep, false_and, ↓reduceIte, List.append_nil]
    apply BlockContinues.branch (flag := false) (by simpa [keep] using tested) rfl
    exact .nil

private theorem polygonal_call (s n : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (globals : Mutable.Globals) :
    Evaluates program env (.call (.name "polygonal") [.name "s", .name "n"]) heap
      (number (polygonal s n)) heap globals :=
  Evaluates.call (.cons (Evaluates.lookup getS) (.cons (Evaluates.lookup getN) .nil))
    (solvers.«61».polygonal.heap_result s n heap globals)

private theorem loop_done (s n : Int) (done : 10000 ≤ polygonal s n)
    (env : HeapCalls.Env) (base : Mutable.Heap) (before : List Int)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals) :
    BlockReturns program env [statement, .return (.name "nums")] (working base before)
      (.ref base.length) (working base before) globals := by
  apply BlockReturns.while_loop (flag := true) (middle := working base before) Evaluates.constant rfl
  apply BlockReturns.assign (polygonal_call s n env (working base before) getS getN globals)
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := working base before)
    (Evaluates.compare (middle := working base before) (afterRight := working base before)
      (av := number (polygonal s n)) (bv := number 10000)
      (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, done])) rfl
  apply BlockReturns.break_loop (next := statement) (following := [.return (.name "nums")]) rfl
  exact BlockReturns.return_value (Evaluates.lookup (by simpa [setLocal] using getNums))

/-- Every actual loop call reaches its break by index 141 and preserves the caller heap prefix. -/
theorem loop (fuel : Nat) (s n : Int) (sides : 3 ≤ s) (positive : 1 ≤ n)
    (bound : 141 ≤ n + fuel) (env : HeapCalls.Env) (base : Mutable.Heap) (before : List Int)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals := []) :
    BlockReturns program env [statement, .return (.name "nums")] (working base before)
      (.ref base.length) (working base (before ++ collect fuel s n)) globals := by
  induction fuel generalizing n env before with
  | zero =>
      have done := ProjectEulerSolutions.PolygonalFamilies.above_four_digits
        s n sides (by simpa using bound)
      simpa [collect] using loop_done s n done env base before getS getN getNums globals
  | succ fuel ih =>
      by_cases done : 10000 ≤ polygonal s n
      · simpa [collect, done] using loop_done s n done env base before getS getN getNums globals
      · let valued := setLocal env "val" (number (polygonal s n))
        let nextEnv := setLocal valued "n" (number (n + 1))
        have valuedS : valued "s" = some (number s) := by simpa [valued, setLocal] using getS
        have valuedN : valued "n" = some (number n) := by simpa [valued, setLocal] using getN
        have valuedNums : valued "nums" = some (.ref base.length) := by simpa [valued, setLocal] using getNums
        have nextS : nextEnv "s" = some (number s) := by simpa [nextEnv, setLocal] using valuedS
        have nextN : nextEnv "n" = some (number (n + 1)) := by simp [nextEnv, setLocal]
        have nextNums : nextEnv "nums" = some (.ref base.length) := by simpa [nextEnv, setLocal] using valuedNums
        have filtered := filter_step (polygonal s n) (by omega) valued base before rfl valuedNums globals
        have rest := ih (n + 1) (by omega) (by omega) nextEnv
          (before ++ emitted (polygonal s n)) nextS nextN nextNums
        simp only [collect, if_neg done]
        apply BlockReturns.while_loop (flag := true) (middle := working base before) Evaluates.constant rfl
        apply BlockReturns.assign (polygonal_call s n env (working base before) getS getN globals)
        apply BlockReturns.branch (flag := false) (value := flag false) (middle := working base before)
          (Evaluates.compare (middle := working base before) (afterRight := working base before)
            (av := number (polygonal s n)) (bv := number 10000)
            (Evaluates.lookup rfl) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, done])) rfl
        apply filtered.returns
        apply BlockReturns.augAssign valuedN Evaluates.constant
          (value := number (n + 1)) (afterOperation := working base (before ++ emitted (polygonal s n)))
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
        · apply BlockReturns.loop_resume
          simpa [nextEnv, List.append_assoc] using rest

/-- The helper returns the complete four-digit family and preserves the caller heap prefix. -/
theorem result (s : Int) (sides : 3 ≤ s) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "generate_4digit_polygonals" [number s] heap (.ref heap.length)
      (working heap (answer s)) globals := by
  let params := ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).2
  let env := functionEnv params sourceBody [number s] globals
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (executableBody sourceBody) heap (.ref heap.length) (working heap (answer s)) globals
  rw [code_source]
  have allocated : BlockContinues program env
      [.annAssign (.name "nums") (.subscript (.name "list") (.name "int")) (some (.list []))]
      heap (setLocal env "nums" (.ref heap.length)) (working heap []) globals := by
    exact BlockContinues.annAssign (Evaluates.list .nil)
  apply allocated.returns
  apply BlockReturns.assign Evaluates.constant
  simpa only [collect_family s sides] using loop 140 s 1 sides (by decide) (by decide)
    (setLocal (setLocal env "nums" (.ref heap.length)) "n" (number 1)) heap []
    (by cases globals <;> rfl) rfl rfl globals

end solvers.«61».generate_4digit_polygonals
