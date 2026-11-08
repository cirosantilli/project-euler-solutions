import «.formalsw».solvers.«18.py.ast»
import lang.py.FormalSW.HeapRuntimeCostComprehensions
import lang.py.FormalSW.HeapRuntimeInputs

namespace solvers.«18».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open HeapCalls.Runtime.NestedIntList (encode)
private abbrev program := Solvers.P18Ast.module

def rowComprehension : Expr :=
  match executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 with
  | [_, .forIn _ _ [_, .assign _ expression] _, _] => expression
  | _ => .constant .none

def rowElement : Expr :=
  match rowComprehension with | .listComp element _ => element | _ => .constant .none

private theorem prefix_read (heap extra : Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).choose]
  exact found

private theorem read_cost (heap : Heap) (env : HeapCalls.Env) (name : String)
    (address position : Nat) (values : List Int) (index : Expr)
    (found : heap[address]? = some (.list (encode values))) (valid : position < values.length)
    (lookup : env name = some (.ref address))
    (evaluated : EvaluatesIn .word program env index heap cost (.scalar (.int position)) heap)
    (ordinary : ∀ a b c, index ≠ .slice a b c) :
    EvaluatesIn .word program env (.subscript (.name name) index) heap
      (cost+9) (.scalar (.int values[position])) heap := by
  have accessed := EvaluatesIn.list_subscript ordinary (EvaluatesIn.lookup lookup) evaluated found
    (Mutable.getItem_list heap address position (encode values) found (by simpa using valid))
  simpa using accessed.mono (show 1+cost+1+1+6 ≤ cost+9 by omega)

theorem element_cost (heap : Heap) (env : HeapCalls.Env)
    (rowAddress belowAddress position : Nat) (row below : List Int)
    (getRow : heap[rowAddress]? = some (.list (encode row)))
    (getBelow : heap[belowAddress]? = some (.list (encode below)))
    (inRow : position < row.length) (inNext : position+1 < below.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref belowAddress))
    (localIndex : env "i" = some (.scalar (.int position))) :
    EvaluatesIn .word program env rowElement heap 53
      (.scalar (.int (row[position] + max below[position] below[position+1]))) heap := by
  have next : EvaluatesIn .word program env (.binOp (.name "i") .add (.constant (.int 1))) heap
      6 (.scalar (.int (position+1))) heap :=
    EvaluatesIn.binary (op := .add) (work := 1) (EvaluatesIn.lookup localIndex) EvaluatesIn.constant rfl rfl
  apply EvaluatesIn.binary (op := .add) (av := .int row[position])
    (bv := .int (max below[position] below[position+1])) (work := 1) (ca := 10) (cb := 39) (afterRight := heap)
  · exact read_cost heap env "row" rowAddress position row (.name "i") getRow inRow localRow
      (EvaluatesIn.lookup localIndex) (by intros a b c impossible; cases impossible)
  · apply EvaluatesIn.maximum_pair (work := 9) (ca := 10) (cb := 15)
      (read_cost heap env "dp" belowAddress position below (.name "i") getBelow (by omega) localBelow
        (EvaluatesIn.lookup localIndex) (by intros a b c impossible; cases impossible))
      (read_cost heap env "dp" belowAddress (position+1) below _ getBelow inNext localBelow next
        (by intros a b c impossible; cases impossible))
    · by_cases larger : below[position] < below[position+1]
      · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare,
          evalCmpOp, HeapCalls.scalarResult, larger]
        omega
      · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare,
          evalCmpOp, HeapCalls.scalarResult, larger]
        omega
    · rfl
  · rfl
  · rfl

private def rowBody (address : Nat) : List Stmt := [.comprehensionAppend address rowElement]

theorem row_loop (remaining position : Nat) (row : List Int) (endpoint : position+remaining = row.length)
    (heap : Heap) (env : HeapCalls.Env) (rowAddress belowAddress : Nat) (below output : List Int)
    (getRow : heap[rowAddress]? = some (.list (encode row)))
    (getBelow : heap[belowAddress]? = some (.list (encode below))) (complete : row.length < below.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref belowAddress)) :
    ∃ finalEnv result, result.length = output.length+remaining ∧
      BlockContinuesIn .word program env
        [.rangeStepLoop "i" position row.length 1 (rowBody heap.length) []]
        (heap ++ [.list (encode output)]) (64*remaining+4) finalEnv (heap ++ [.list (encode result)]) := by
  induction remaining generalizing position env output with
  | zero =>
      refine ⟨env, output, by simp, ?_⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
  | succ remaining ih =>
      let indexed := setLocal env "i" (.scalar (.int position))
      let value := row[position]'(by omega) + max (below[position]'(by omega)) (below[position+1]'(by omega))
      have savedRow : indexed "row" = some (.ref rowAddress) := by simpa [indexed, setLocal] using localRow
      have savedBelow : indexed "dp" = some (.ref belowAddress) := by simpa [indexed, setLocal] using localBelow
      have emitted : BlockContinuesIn .word program indexed (rowBody heap.length)
          (heap ++ [.list (encode output)]) 60 indexed (heap ++ [.list (encode (output ++ [value]))]) := by
        have run := BlockContinuesIn.comprehension_append
          (element_cost _ indexed rowAddress belowAddress position row below
            (prefix_read heap _ _ _ getRow) (prefix_read heap _ _ _ getBelow) (by omega) (by omega) savedRow savedBelow rfl)
          (by simp : (heap ++ [.list (encode output)])[heap.length]? = some (.list (encode output)))
        simpa [rowBody, encode, value, List.map_append] using run
      obtain ⟨finalEnv, result, size, tail⟩ := ih (position+1) (by omega) indexed (output ++ [value]) savedRow savedBelow
      refine ⟨finalEnv, result, by simp only [List.length_append, List.length_singleton] at size; omega, ?_⟩
      have continued := BlockContinuesIn.range_step (env := env) (name := "i")
        (next := position) (stop := row.length) (stride := 1) (body := rowBody heap.length) (otherwise := [])
        (by simp [Calls.rangeContinues]; omega)
        (by simpa [rowBody, Calls.loopContinuation, Int.natCast_add] using emitted.append tail)
      exact continued.mono (by omega)

/-- The actual source comprehension creates one result per row entry. Its
cost is independent of integer magnitudes under the fixed-arithmetic model. -/
theorem row_cost (heap : Heap) (env : HeapCalls.Env) (rowAddress belowAddress : Nat) (row below : List Int)
    (getRow : heap[rowAddress]? = some (.list (encode row)))
    (getBelow : heap[belowAddress]? = some (.list (encode below))) (complete : row.length < below.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref belowAddress)) :
    ∃ result, result.length = row.length ∧ EvaluatesIn .word program env rowComprehension heap
      (64*row.length+17) (.ref heap.length) (heap ++ [.list (encode result)]) := by
  let generators : List (String × Expr × List Expr) :=
    [("i", .call (.name "range") [.builtinCall .len [.name "row"]], [])]
  let masked := comprehensionEnv env generators
  obtain ⟨finalEnv, result, size, loop⟩ := row_loop row.length 0 row (by omega) heap masked rowAddress belowAddress below []
    getRow getBelow complete
    (by simpa [masked, generators, comprehensionEnv] using localRow)
    (by simpa [masked, generators, comprehensionEnv] using localBelow)
  have length : EvaluatesIn .word program env (.builtinCall .len [.name "row"]) heap 5
      (.scalar (.int row.length)) heap := by
    apply EvaluatesIn.builtin_single (builtin := .len) (work := 1)
      (by intro h; cases h) (by intro h; cases h) (EvaluatesIn.lookup localRow)
    · simp [HeapCalls.primitive, Mutable.length, getRow]
    · simp [operation, sequenceLength, getRow]
  refine ⟨result, by simpa using size, ?_⟩
  have run := EvaluatesIn.list_comp_range (work := 3) (start := 0) (stop := row.length) (stride := 1)
    (name := "i") (element := rowElement) (filters := []) (generators := []) (cb := 64*row.length+4)
    (ArgumentsIn.cons length ArgumentsIn.nil) rfl rfl
    (by simpa [rowBody, comprehensionFilters, comprehensionBody] using loop)
  exact run.mono (by omega)

end solvers.«18».runtimeCost
