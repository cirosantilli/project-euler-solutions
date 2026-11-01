import solvers.«74.py.lean.d».Heap

/-! The actual forward walk stops at a cached tail or the first repeated path position. -/
namespace solvers.«74».walk
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.BoundedOrbit ProjectEulerStatements.DigitFactorialChains
set_option Elab.async false

abbrev program := Solvers.P74Ast.module
abbrev number := factorials.number
abbrev condition : Expr := .boolOp .and
  [.membership (.name "current") (.name "lengths") true,
   .membership (.name "current") (.name "positions") true]
abbrev body : List Stmt :=
  [.assign [.subscript (.name "positions") (.name "current")]
      (.builtinCall .len [.name "path"]),
   .expr (.call (.attribute (.name "path") "append") [.name "current"]),
   .assign [.name "current"] (.call (.name "next_term") [.name "current"])]
abbrev loopCode : List Stmt := [.whileDo condition body []]

theorem code_source :
    (executableBody ((findFunctionIn program.body "chain_length").getD ([], [])).2)[3]? =
      some (.whileDo condition body []) := rfl

structure Captured (locals : HeapCalls.Env) (base : Mutable.Heap)
    (start cache current : Nat) : Prop where
  startValue : locals "start" = some (number start)
  cacheValue : locals "lengths" = some (.ref cache)
  pathValue : locals "path" = some (.ref base.length)
  positionsValue : locals "positions" = some (.ref (base.length + 1))
  currentValue : locals "current" = some (number current)

theorem Captured.advance {locals : HeapCalls.Env} {base : Mutable.Heap}
    {start cache current : Nat} (captured : Captured locals base start cache current) :
    Captured (setLocal locals "current" (number (nextTerm current)))
      base start cache (nextTerm current) where
  startValue := by simpa [setLocal] using captured.startValue
  cacheValue := by simpa [setLocal] using captured.cacheValue
  pathValue := by simpa [setLocal] using captured.pathValue
  positionsValue := by simpa [setLocal] using captured.positionsValue
  currentValue := by simp [setLocal]

structure Prefix (start current : Nat) (path : List Nat) : Prop where
  visited : path = segment nextTerm start path.length
  frontier : nextTerm^[path.length] start = current

theorem Prefix.initial (start : Nat) : Prefix start start [] := ⟨rfl, rfl⟩

theorem Prefix.advance {start current : Nat} {path : List Nat}
    (prefixPath : Prefix start current path) :
    Prefix start (nextTerm current) (path ++ [current]) := by
  constructor
  · simp only [List.length_append, List.length_singleton]
    calc
      path ++ [current] = segment nextTerm start path.length ++ [current] :=
        congrArg (fun values => values ++ [current]) prefixPath.visited
      _ = segment nextTerm start path.length ++
          segment nextTerm (nextTerm^[path.length] start) 1 := by
            simp [segment, prefixPath.frontier]
      _ = segment nextTerm start (path.length + 1) :=
        (segment_append nextTerm start path.length 1).symm
  · simp only [List.length_append, List.length_singleton, Function.iterate_succ_apply',
      prefixPath.frontier]

def Uncached (cache : store.Entries) (path : List Nat) : Prop :=
  ∀ n ∈ path, store.lookup cache n = none

theorem Uncached.advance {cache : store.Entries} {path : List Nat} {current : Nat}
    (uncached : Uncached cache path) (missing : store.lookup cache current = none) :
    Uncached cache (path ++ [current]) := by
  intro n member
  rcases List.mem_append.mp member with earlier | last
  · exact uncached n earlier
  · simpa only [List.mem_singleton.mp last] using missing

inductive Stopped (cache positions : store.Entries) (current : Nat) : Prop
  | cached (length : Nat) :
      store.lookup cache current = some length → Stopped cache positions current
  | cycle (index : Nat) : store.lookup cache current = none →
      store.lookup positions current = some index → Stopped cache positions current

structure Outcome (base : Mutable.Heap) (start cacheAddress : Nat)
    (cache : store.Entries) (whole : List Nat) where
  current : Nat
  path : List Nat
  positions : store.Entries
  garbage : Mutable.Heap
  locals : HeapCalls.Env
  captured : Captured locals base start cacheAddress current
  recorded : solvers.«74».positions.Exact positions path
  visited : Prefix start current path
  uncached : Uncached cache path
  included : path <+: whole
  cycleComplete : store.lookup cache current = none → path = whole
  stopped : Stopped cache positions current

variable {globals : Mutable.Globals}

theorem guard_result (base : Mutable.Heap) (cache : store.Entries)
    (cacheAddress start current : Nat)
    (path : List Nat) (positions : store.Entries) (garbage : Mutable.Heap)
    (locals : HeapCalls.Env) (captured : Captured locals base start cacheAddress current)
    (found : store.Found base cacheAddress cache) :
    Evaluates program locals condition (heap.work base path positions garbage)
      (.scalar (.bool (!(store.lookup cache current).isSome &&
        !(store.lookup positions current).isSome)))
      (heap.work base path positions garbage) globals := by
  have left := store.membership_evaluates (globals := globals)
    (heap.work base path positions garbage) locals "lengths" "current" cacheAddress current cache
    (heap.old_found base path positions garbage cacheAddress _ found)
    captured.cacheValue captured.currentValue true
  have right := store.membership_evaluates (globals := globals)
    (heap.work base path positions garbage) locals "positions" "current" (base.length + 1)
    current positions (heap.positions_found base path positions garbage)
    captured.positionsValue captured.currentValue true
  cases selected : (store.lookup cache current).isSome with
  | false =>
      apply Evaluates.boolean_next (flag := true)
        (by simpa only [selected, Bool.not_false, Bool.true_and] using left) rfl rfl
      exact Evaluates.boolean_last right rfl
  | true =>
      apply Evaluates.boolean_stop (flag := false)
        (by simpa only [selected, Bool.not_true, Bool.false_and] using left) rfl rfl

theorem iteration (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage : Mutable.Heap) (locals : HeapCalls.Env) (start cache current address : Nat)
    (captured : Captured locals base start cache current)
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (found : base[address]? = some (.list factorials.values)) :
    ∃ extra, BlockContinues program locals body (heap.work base path positions garbage)
      (setLocal locals "current" (number (nextTerm current)))
      (heap.work base (path ++ [current]) (store.insert positions current path.length)
        (garbage ++ extra)) globals := by
  have written := heap.write_position (globals := globals)
    base path positions garbage locals current
    captured.pathValue captured.positionsValue captured.currentValue
  have appended := BlockContinues.discard
    (heap.path_append (globals := globals) base path (store.insert positions current path.length)
      garbage locals current captured.pathValue captured.currentValue)
  obtain ⟨extra, evaluated⟩ := heap.successor base (path ++ [current])
    (store.insert positions current path.length) garbage locals current address
    captured.currentValue getTable found
  refine ⟨extra, ?_⟩
  exact written.append (appended.append (BlockContinues.assign evaluated))

private theorem stopped_result (start current cacheAddress : Nat) (path whole : List Nat)
    (trace : Trace nextTerm current path whole) (base : Mutable.Heap)
    (cache pos : store.Entries) (garbage : Mutable.Heap) (locals : HeapCalls.Env)
    (found : store.Found base cacheAddress cache)
    (captured : Captured locals base start cacheAddress current)
    (recorded : positions.Exact pos path) (visited : Prefix start current path)
    (uncached : Uncached cache path) (stopped : Stopped cache pos current) :
    ∃ finished : Outcome base start cacheAddress cache whole,
      BlockContinues program locals loopCode (heap.work base path pos garbage)
        finished.locals (heap.work base finished.path finished.positions finished.garbage)
        globals := by
  have complete : store.lookup cache current = none → path = whole := by
    intro missing
    cases trace with
    | done => rfl
    | @step current path whole fresh rest =>
        cases stopped with
        | cached length selected => simp [missing] at selected
        | cycle index _ selected =>
            exact False.elim (fresh (List.mem_of_getElem? (recorded.sound current index selected)))
  let finished : Outcome base start cacheAddress cache whole :=
    ⟨current, path, pos, garbage, locals, captured, recorded, visited, uncached,
      trace.prefix, complete, stopped⟩
  refine ⟨finished, ?_⟩
  have tested := guard_result (globals := globals) base cache cacheAddress start current path pos
    garbage locals captured found
  have stoppedGuard : Evaluates program locals condition (heap.work base path pos garbage)
      (.scalar (.bool false)) (heap.work base path pos garbage) globals := by
    cases stopped with
    | cached length selected =>
        simpa only [selected, Option.isSome_some, Bool.not_true, Bool.false_and] using tested
    | cycle index missing selected =>
        simpa only [missing, selected, Option.isSome_none, Option.isSome_some,
          Bool.not_false, Bool.not_true, Bool.true_and] using tested
  apply BlockContinues.while_loop stoppedGuard rfl
  exact .nil

/-- The full mathematical trace supplies termination even when the cache stops the walk early. -/
theorem loop (start current cacheAddress address : Nat) (path whole : List Nat)
    (trace : Trace nextTerm current path whole) (base : Mutable.Heap)
    (cache pos : store.Entries) (garbage : Mutable.Heap) (locals : HeapCalls.Env)
    (foundCache : store.Found base cacheAddress cache)
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (foundTable : base[address]? = some (.list factorials.values))
    (captured : Captured locals base start cacheAddress current)
    (recorded : positions.Exact pos path) (visited : Prefix start current path)
    (uncached : Uncached cache path) :
    ∃ finished : Outcome base start cacheAddress cache whole,
      BlockContinues program locals loopCode (heap.work base path pos garbage)
        finished.locals (heap.work base finished.path finished.positions finished.garbage)
        globals := by
  induction trace generalizing pos garbage locals with
  | @done current path present =>
      cases cached : store.lookup cache current with
      | some length =>
          exact stopped_result start current cacheAddress path path (.done present) base cache pos
            garbage locals foundCache captured recorded visited uncached (.cached length cached)
      | none =>
          cases selected : store.lookup pos current with
          | none =>
              exact False.elim
                ((positions.missing_iff pos path recorded current).mp selected present)
          | some index =>
              exact stopped_result start current cacheAddress path path (.done present)
                base cache pos garbage locals foundCache captured recorded visited uncached
                (.cycle index cached selected)
  | @step current path whole fresh rest ih =>
      cases cached : store.lookup cache current with
      | some length =>
          exact stopped_result start current cacheAddress path whole (.step fresh rest)
            base cache pos
            garbage locals foundCache captured recorded visited uncached (.cached length cached)
      | none =>
          have missing := (positions.missing_iff pos path recorded current).mpr fresh
          obtain ⟨extra, advanced⟩ := iteration base path pos garbage locals start cacheAddress
            current address captured getTable foundTable
          obtain ⟨finished, completed⟩ := ih (store.insert pos current path.length)
            (garbage ++ extra)
            (setLocal locals "current" (number (nextTerm current))) captured.advance
            (positions.insert pos path recorded current fresh) visited.advance
            (uncached.advance cached)
          refine ⟨finished, ?_⟩
          have tested := guard_result (globals := globals) base cache cacheAddress start current
            path pos garbage locals captured foundCache
          apply BlockContinues.while_loop (flag := true)
            (by simpa only [cached, missing, Option.isSome_none, Bool.not_false, Bool.true_and]
              using tested) rfl
          simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            Bool.false_or, Bool.false_eq_true, if_false] using advanced.append completed

abbrev initialization : List Stmt :=
  [.assign [.name "path"] (.list []), .assign [.name "positions"] (.dict []),
   .assign [.name "current"] (.name "start")]

theorem head_source :
    (executableBody ((findFunctionIn program.body "chain_length").getD ([], [])).2).take 4 =
      initialization ++ loopCode := rfl

theorem initialize_locals (base : Mutable.Heap) (locals : HeapCalls.Env) (start cacheAddress : Nat)
    (getStart : locals "start" = some (number start))
    (getCache : locals "lengths" = some (.ref cacheAddress)) :
    ∃ ready, BlockContinues program locals initialization base ready (heap.work base [] [] [])
      globals ∧ Captured ready base start cacheAddress start := by
  let pathEnv := setLocal locals "path" (.ref base.length)
  let posEnv := setLocal pathEnv "positions" (.ref (base.length + 1))
  let ready := setLocal posEnv "current" (number start)
  have first : BlockContinues program locals [.assign [.name "path"] (.list [])]
      base pathEnv (base ++ [.list []]) globals :=
    BlockContinues.assign (Evaluates.list .nil)
  have second : BlockContinues program pathEnv [.assign [.name "positions"] (.dict [])]
      (base ++ [.list []]) posEnv (base ++ [.list [], .dict []]) globals := by
    have allocated := Evaluates.dictionary (globals := globals) (module := program) (env := pathEnv)
      (heap := base ++ [.list []]) .nil
    simpa only [posEnv, List.length_append, List.length_singleton, List.append_assoc,
      List.cons_append, List.nil_append] using BlockContinues.assign allocated
  have third : BlockContinues program posEnv [.assign [.name "current"] (.name "start")]
      (base ++ [.list [], .dict []]) ready (base ++ [.list [], .dict []]) globals :=
    BlockContinues.assign (Evaluates.lookup (by simpa [posEnv, pathEnv, setLocal] using getStart))
  refine ⟨ready, ?_, ?_⟩
  · simpa only [initialization, heap.work, heap.numbers, List.map_nil] using
      first.append (second.append third)
  · constructor
    · simpa [ready, posEnv, pathEnv, setLocal] using getStart
    · simpa [ready, posEnv, pathEnv, setLocal] using getCache
    · simp [ready, posEnv, pathEnv, setLocal]
    · simp [ready, posEnv, setLocal]
    · simp [ready, setLocal]

/-- The complete source prefix allocates its local objects and runs the actual unbounded walk. -/
theorem forward_result (start cacheAddress address : Nat) (base : Mutable.Heap)
    (cache : store.Entries) (locals : HeapCalls.Env)
    (getStart : locals "start" = some (number start))
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (foundCache : store.Found base cacheAddress cache)
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (foundTable : base[address]? = some (.list factorials.values)) :
    ∃ finished : Outcome base start cacheAddress cache (path start),
      BlockContinues program locals
        ((executableBody ((findFunctionIn program.body "chain_length").getD ([], [])).2).take 4)
        base finished.locals (heap.work base finished.path finished.positions finished.garbage)
        globals := by
  obtain ⟨ready, initialized, captured⟩ :=
    initialize_locals base locals start cacheAddress getStart getCache
  obtain ⟨finished, completed⟩ := loop start start cacheAddress address [] (path start)
    (path_trace start) base cache [] [] ready foundCache getTable foundTable captured
    positions.empty (Prefix.initial start) (by simp [Uncached])
  refine ⟨finished, ?_⟩
  rw [head_source]
  exact initialized.append completed

end solvers.«74».walk
