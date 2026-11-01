import solvers.«74.py.lean.d».Chains
import ProjectEulerStatements.P74

/-! The actual counting loop shares one correct cache across every positive starting value. -/
namespace solvers.«74».solve
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.DigitFactorialChains
set_option Elab.async false

abbrev program := Solvers.P74Ast.module
abbrev number := factorials.number
abbrev integer (n : Int) : Mutable.Value := .scalar (.int n)
abbrev test : Expr := .compare (.call (.name "chain_length") [.name "start", .name "lengths"])
  [(.eq, .name "target")]
abbrev body : List Stmt :=
  [.ifThen test [.augAssign (.name "count") .add (.constant (.int 1))] []]

def picked (target : Int) (first count : Nat) : Nat :=
  ((List.range' first count).filter (fun n => decide ((chainLen n : Int) = target))).length

def work (caller : Mutable.Heap) (entries : store.Entries)
    (garbage : Mutable.Heap) : Mutable.Heap :=
  caller ++ (.dict (store.encode entries) :: garbage)

theorem cache_found (caller : Mutable.Heap) (entries : store.Entries) (garbage : Mutable.Heap) :
    store.Found (work caller entries garbage) caller.length entries := by simp [store.Found, work]

theorem old_found (caller : Mutable.Heap) (entries : store.Entries) (garbage : Mutable.Heap)
    (address : Nat) (object : Mutable.Object) (found : caller[address]? = some object) :
    (work caller entries garbage)[address]? = some object :=
  (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found

theorem set_cache (caller : Mutable.Heap) (entries replacement : store.Entries)
    (garbage extra : Mutable.Heap) :
    (work caller entries garbage).set caller.length (.dict (store.encode replacement)) ++ extra =
      work caller replacement (garbage ++ extra) := by
  simp [work, List.set_append_right, List.append_assoc]

variable {globals : Mutable.Globals}

theorem tested (start address : Nat) (target : Int) (caller garbage : Mutable.Heap)
    (entries : store.Entries) (valid : memo.Valid entries) (locals : HeapCalls.Env)
    (getStart : locals "start" = some (number start))
    (getCache : locals "lengths" = some (.ref caller.length))
    (getTarget : locals "target" = some (integer target))
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (foundTable : caller[address]? = some (.list factorials.values)) :
    ∃ finalEntries extra, memo.Valid finalEntries ∧
      Evaluates program locals test (work caller entries garbage)
        (.scalar (.bool (decide ((chainLen start : Int) = target))))
        (work caller finalEntries (garbage ++ extra)) globals := by
  obtain ⟨finalEntries, extra, finalValid, _, returned⟩ := chain_length.result (globals := globals)
    start caller.length address (work caller entries garbage) entries valid
    (cache_found caller entries garbage) getTable
    (old_found caller entries garbage address _ foundTable)
  refine ⟨finalEntries, extra, finalValid, ?_⟩
  rw [set_cache] at returned
  exact Evaluates.compare
    (Evaluates.call (.cons (Evaluates.lookup getStart) (.cons (Evaluates.lookup getCache) .nil))
      returned) (Evaluates.lookup getTarget) rfl

theorem scan (target stop : Int) (first count total address : Nat)
    (length : (stop - (first : Int)).toNat = count)
    (caller garbage : Mutable.Heap) (entries : store.Entries) (valid : memo.Valid entries)
    (locals : HeapCalls.Env) (getCache : locals "lengths" = some (.ref caller.length))
    (getTarget : locals "target" = some (integer target))
    (getTotal : locals "count" = some (number total))
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (foundTable : caller[address]? = some (.list factorials.values)) :
    ∃ finalEntries extra, memo.Valid finalEntries ∧
      BlockReturns program locals
        [.rangeStepLoop "start" first stop 1 body [], .return (.name "count")]
        (work caller entries garbage) (number (total + picked target first count))
        (work caller finalEntries (garbage ++ extra)) globals := by
  induction count generalizing first total garbage entries locals with
  | zero =>
      refine ⟨entries, [], valid, ?_⟩
      simp only [picked, List.range'_zero, List.filter_nil, List.length_nil,
        Nat.add_zero, List.append_nil]
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | succ count ih =>
      let entered := setLocal locals "start" (number first)
      have cacheValue : entered "lengths" = some (.ref caller.length) := by
        simpa [entered, setLocal] using getCache
      have targetValue : entered "target" = some (integer target) := by
        simpa [entered, setLocal] using getTarget
      have countValue : entered "count" = some (number total) := by
        simpa [entered, setLocal] using getTotal
      obtain ⟨middleEntries, before, middleValid, evaluated⟩ := tested (globals := globals)
        first address target caller garbage entries valid entered (by simp [entered, setLocal])
        cacheValue targetValue getTable foundTable
      by_cases chosen : (chainLen first : Int) = target
      · let updated := setLocal entered "count" (number (total + 1))
        obtain ⟨finalEntries, extra, finalValid, rest⟩ := ih (first + 1) (total + 1)
          (by omega) (garbage ++ before) middleEntries middleValid updated
          (by simpa [updated, setLocal] using cacheValue)
          (by simpa [updated, setLocal] using targetValue) (by simp [updated, setLocal])
        refine ⟨finalEntries, before ++ extra, finalValid, ?_⟩
        apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
        apply BlockReturns.branch (flag := true)
          (by simpa only [chosen, decide_true] using evaluated) rfl
        apply BlockReturns.augAssign countValue Evaluates.constant
          (value := number (total + 1))
          (afterOperation := work caller middleEntries (garbage ++ before))
        · simp [inplace, Mutable.binary, scalarBinary, number, factorials.number]
        · simpa [picked, List.range'_succ, chosen, body, Calls.loopContinuation,
            Calls.loopControl, Calls.loopControlStmt, entered, updated, List.append_assoc,
            Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using rest
      · obtain ⟨finalEntries, extra, finalValid, rest⟩ := ih (first + 1) total
          (by omega) (garbage ++ before) middleEntries middleValid entered
          cacheValue targetValue countValue
        refine ⟨finalEntries, before ++ extra, finalValid, ?_⟩
        apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
        apply BlockReturns.branch (flag := false)
          (by simpa only [chosen, decide_false] using evaluated) rfl
        simpa [picked, List.range'_succ, chosen, body, Calls.loopContinuation,
          Calls.loopControl, Calls.loopControlStmt, entered, List.append_assoc] using rest

/-- Arbitrary signed bounds and targets execute the original solver and preserve caller objects. -/
theorem result (limit target : Int) (caller : Mutable.Heap) (address : Nat)
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (foundTable : caller[address]? = some (.list factorials.values)) :
    ∃ extra, ReturnsValue program "solve" [integer limit, integer target] caller
      (integer (ProjectEulerStatements.P74.result limit target)) (caller ++ extra) globals := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let source := ((findFunctionIn program.body "solve").getD ([], [])).2
  let entry := functionEnv args source [integer limit, integer target] globals
  let ready := setLocal (setLocal entry "lengths" (.ref caller.length)) "count" (number 0)
  have getLimit : ready "limit" = some (integer limit) := by cases globals <;> rfl
  have getTarget : ready "target" = some (integer target) := by cases globals <;> rfl
  obtain ⟨finalEntries, extra, _, scanned⟩ := scan (globals := globals)
    target limit 1 (limit - 1).toNat 0 address rfl caller [] [] Memo.empty_valid ready
    (by simp [ready, setLocal]) getTarget (by simp [ready, setLocal]) getTable foundTable
  refine ⟨.dict (store.encode finalEntries) :: extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (Evaluates.dictionary .nil)
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range
    (.cons Evaluates.constant (.cons (Evaluates.lookup getLimit) .nil)) rfl
  simpa only [work, List.nil_append, picked, Nat.zero_add, ProjectEulerStatements.P74.result]
    using scanned

theorem initialized_result (limit target : Int) (caller : Mutable.Heap) :
    ∃ extra, ReturnsValue program "solve" [integer limit, integer target] (factorials.heap caller)
      (integer (ProjectEulerStatements.P74.result limit target)) (factorials.heap caller ++ extra)
      (factorials.globals caller) :=
  result limit target (factorials.heap caller) caller.length (by simp [factorials.globals])
    (factorials.found caller)

end solvers.«74».solve
