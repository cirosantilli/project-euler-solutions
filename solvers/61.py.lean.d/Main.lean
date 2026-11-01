import solvers.«61.py.lean.d».Starts

set_option Elab.async false

namespace solvers.«61».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open dfs (numbers)
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def answer (maxSides : Int) : Int :=
  PolygonalCycleSearch.optimized PolygonalFamilies.family (PolygonalCycleSearch.indexed PolygonalFamilies.family) maxSides

def initialTypeValues (maxSides : Nat) : List Int :=
  (List.range' 3 (maxSides - 2)).map (fun (kind : Nat) => (kind : Int))
def initialTypesExpression : Expr :=
  .listComp (.name "$formalsw.range.item")
    [("$formalsw.range.item", .call (.name "range")
      [.constant (.int 3), .binOp (.name "max_sides") .add (.constant (.int 1))], [])]

def sourceCode : List Stmt :=
  [.ifThen (.compare (.name "max_sides") [(.lt, .constant (.int 3))]) [.return (.constant (.int 0))] [],
   .assign [.name "types"] initialTypesExpression,
   .assign [.name "nums_by_type"] tables.expression,
   .assign [.name "__formalsw_cell_solve_pref_map"] (.list [prefix_setup.expression]),
   .assign [.name "best"] (.constant (.int 0)),
   .forIn (.name "start") startIterable startBody [], .return (.name "best")]

theorem code_source :
    executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 = sourceCode := rfl

theorem types_result (maxSides : Nat) (large : 3 ≤ maxSides) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getMax : env "max_sides" = some (number maxSides)) (globals : Mutable.Globals := []) :
    ∃ extra, Evaluates program env initialTypesExpression heap (.ref heap.length)
      (heap ++ (.list (numbers (initialTypeValues maxSides)) :: extra)) globals := by
  obtain ⟨extra, collected⟩ := HeapCollection.range_map (globals := globals) program "$formalsw.range.item"
    (.name "$formalsw.range.item") (fun (n : Nat) => number n)
    (fun heap env n found => ⟨[], by simpa using
      (Evaluates.lookup (module := program) (heap := heap) (globals := globals) found)⟩)
    heap env [.constant (.int 3), .binOp (.name "max_sides") .add (.constant (.int 1))]
    [number 3, number (maxSides + 1)] 3 (maxSides - 2)
    (.cons Evaluates.constant (.cons (Evaluates.binary (av := number maxSides) (bv := number 1)
      (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup getMax) Evaluates.constant (by simp [Mutable.binary, scalarBinary])) .nil))
    (by change Except.ok (ε := ExceptionObject) ((3 : Int), ((maxSides + 1 : Nat) : Int), (1 : Int)) =
          .ok (3, (3 : Int) + ((maxSides - 2 : Nat) : Int), 1)
        have endpoint : ((maxSides + 1 : Nat) : Int) = (3 : Int) + ((maxSides - 2 : Nat) : Int) := by omega
        rw [endpoint])
  refine ⟨extra, ?_⟩
  simpa only [initialTypesExpression, HeapCollection.working, initialTypeValues, numbers,
    List.map_map, Function.comp_def] using collected

/-- All setup callbacks and the full indexed recursion return the complete maximum-search model. -/
theorem positive_result (maxSides : Nat) (large : 3 ≤ maxSides) (caller : Mutable.Heap)
    (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "solve" [number maxSides] caller (number (answer maxSides)) (caller ++ extra) globals := by
  let params := ((findFunctionIn program.body "solve").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := functionEnv params sourceBody [number maxSides] globals
  have getMax : env "max_sides" = some (number maxSides) := by cases globals <;> rfl
  obtain ⟨typesGarbage, typesBuilt⟩ := types_result maxSides large caller env getMax globals
  let typedHeap := caller ++ (.list (numbers (initialTypeValues maxSides)) :: typesGarbage)
  let typedEnv := setLocal env "types" (.ref caller.length)
  have typesFound : typedHeap[caller.length]? = some (.list (numbers (initialTypeValues maxSides))) := by simp [typedHeap]
  have valid : ∀ kind ∈ initialTypeValues maxSides, (3 : Int) ≤ kind := by
    intro kind member
    obtain ⟨index, inRange, equal⟩ := List.mem_map.mp member
    have bound := List.mem_range'_1.mp inRange
    omega
  obtain ⟨numsHeap, numsStore, numbersBuilt⟩ := tables.result (initialTypeValues maxSides) valid typedHeap
    caller.length typesFound typedEnv rfl globals
  let numsEnv := setLocal typedEnv "nums_by_type" (.ref typedHeap.length)
  have numsTypesFound := tables.source_found numsStore caller.length (initialTypeValues maxSides) typesFound
  obtain ⟨prefHeap, prefStore, prefixesBuilt⟩ := prefix_setup.result numsStore caller.length numsTypesFound numsEnv
    (by simp [numsEnv, typedEnv, setLocal]) (by simp [numsEnv, setLocal]) globals
  have covered : ∀ kind : Int, 3 ≤ kind ∧ kind ≤ maxSides → kind ∈ initialTypeValues maxSides := by
    intro kind bounds
    apply List.mem_map.mpr
    exact ⟨kind.toNat, List.mem_range'_1.mpr ⟨by omega, by omega⟩, by omega⟩
  obtain ⟨world⟩ := prefStore.world maxSides covered
  let captured := setLocal numsEnv "__formalsw_cell_solve_pref_map" (.ref prefHeap.length)
  let ready := setLocal captured "best" (number 0)
  have readyMax : ready "max_sides" = some (number maxSides) := by
    simpa [ready, captured, numsEnv, typedEnv, setLocal] using getMax
  obtain ⟨prefixBuffer, prefixLayout⟩ := prefStore.extension
  have dataExtension : ∃ extra, prefHeap ++ [.list [.ref numsHeap.length]] = numsHeap ++ extra :=
    ⟨prefixBuffer ++ [.list [.ref numsHeap.length]], by rw [prefixLayout, List.append_assoc]⟩
  obtain ⟨searched, finalEnv, executed, finalBest⟩ := start_loop maxSides large world ready
    (by simp [ready, captured, setLocal]) readyMax numsStore (covered 3 ⟨by decide, by omega⟩) dataExtension
    (by simp [ready, captured, numsEnv, setLocal]) 0 rfl globals
  have model : answer maxSides = (PolygonalFamilies.family 3).foldl
      (fun best start => max best (startValue maxSides start)) 0 := by
    rw [answer, PolygonalCycleSearch.optimized, if_neg (by omega : ¬ (maxSides : Int) < 3)]
    rfl
  rw [← model] at finalBest
  obtain ⟨numberBuffer, numberLayout⟩ := numsStore.extension
  let extra := (.list (numbers (initialTypeValues maxSides)) :: typesGarbage) ++ numberBuffer ++
    prefixBuffer ++ [.list [.ref numsHeap.length]] ++ searched
  have finalLayout : (prefHeap ++ [.list [.ref numsHeap.length]]) ++ searched = caller ++ extra := by
    simp only [extra, prefixLayout, numberLayout, typedHeap, List.append_assoc]
  refine ⟨extra, ?_⟩
  rw [← finalLayout]
  apply BlockReturns.function rfl rfl
  change BlockReturns program env sourceCode caller _ _ globals
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := caller)
  · exact Evaluates.compare (av := number maxSides) (bv := number 3) (middle := caller) (afterRight := caller)
      (Evaluates.lookup getMax) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]; omega)
  · rfl
  · apply BlockReturns.assign typesBuilt
    apply BlockReturns.assign numbersBuilt
    apply BlockReturns.assign (Evaluates.list (.cons prefixesBuilt .nil))
    apply BlockReturns.assign Evaluates.constant
    exact executed.returns (BlockReturns.return_value (Evaluates.lookup finalBest))

theorem heap_result (maxSides : Int) (caller : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "solve" [number maxSides] caller (number (answer maxSides)) (caller ++ extra) globals := by
  by_cases low : maxSides < 3
  · refine ⟨[], ?_⟩
    simp only [answer, PolygonalCycleSearch.optimized, if_pos low, List.append_nil]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := true) (value := flag true) (middle := caller)
    · exact Evaluates.compare (av := number maxSides) (bv := number 3) (middle := caller) (afterRight := caller)
        (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, low])
    · rfl
    · exact BlockReturns.return_value Evaluates.constant
  · lift maxSides to Nat using (by omega : 0 ≤ maxSides)
    exact positive_result maxSides (by omega) caller globals

theorem «Returns» (maxSides : Int) :
    (HeapCalls.scalarFunction program "solve" [.int maxSides]).Returns := by
  obtain ⟨extra, returned⟩ := heap_result maxSides []
  exact returned.scalar_returns (values := [.int maxSides])

theorem default_result (caller : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "solve" [] caller (number (answer 8)) (caller ++ extra) globals := by
  obtain ⟨extra, returned⟩ := heap_result 8 caller globals
  exact ⟨extra, returned.with_defaults (supplied := []) rfl rfl rfl⟩

end solvers.«61».solve
