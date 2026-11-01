import solvers.«74.py.lean.d».Memo

/-! The memoized source helper returns the exact length and retains a complete correct cache. -/
namespace solvers.«74».chain_length
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.DigitFactorialChains
set_option Elab.async false

abbrev program := Solvers.P74Ast.module
abbrev number := factorials.number

structure Written (start cacheAddress : Nat) where
  entries : store.Entries
  extra : Mutable.Heap
  locals : HeapCalls.Env
  valid : memo.Valid entries
  selected : store.lookup entries start = some (chainLen start)
  startValue : locals "start" = some (number start)
  cacheValue : locals "lengths" = some (.ref cacheAddress)

variable {globals : Mutable.Globals}

theorem writeback (base : Mutable.Heap) (start cacheAddress : Nat) (entries : store.Entries)
    (valid : memo.Valid entries) (found : store.Found base cacheAddress entries)
    (walked : walk.Outcome base start cacheAddress entries (path start)) :
    ∃ finished : Written start cacheAddress,
      BlockContinues program walked.locals write.code
        (heap.work base walked.path walked.positions walked.garbage) finished.locals
        (base.set cacheAddress (.dict (store.encode finished.entries)) ++ finished.extra)
        globals := by
  have tested := store.membership_evaluates (globals := globals)
    (heap.work base walked.path walked.positions walked.garbage) walked.locals
    "lengths" "current" cacheAddress walked.current entries
    (heap.old_found base walked.path walked.positions walked.garbage cacheAddress _ found)
    walked.captured.cacheValue walked.captured.currentValue false
  have distinct : walked.path.Nodup := (path_distinct start).sublist walked.included.sublist
  cases stopped : walked.stopped with
  | cached remaining selected =>
      obtain ⟨correct, returned⟩ := memo.cached_correct start walked.current remaining walked.path
        entries valid walked.visited distinct walked.uncached selected
      let locals := write.tailLocals (setLocal walked.locals "remaining" (number remaining))
        remaining walked.path.reverse
      let finalEntries := write.tailFill entries remaining walked.path.reverse
      let extra := Mutable.Object.list (walked.path.map number) ::
        .dict (store.encode walked.positions) :: walked.garbage
      let finished : Written start cacheAddress :=
        ⟨finalEntries, extra, locals, correct, returned,
          by simpa [locals, write.tailLocals_frame, setLocal] using walked.captured.startValue,
          by simpa [locals, write.tailLocals_frame, setLocal] using walked.captured.cacheValue⟩
      refine ⟨finished, ?_⟩
      have executed := branches.cached_result (globals := globals)
        (heap.work base walked.path walked.positions walked.garbage) walked.locals
        cacheAddress base.length walked.current remaining entries walked.path
        (heap.old_found base walked.path walked.positions walked.garbage cacheAddress _ found)
        (heap.path_found base walked.path walked.positions walked.garbage)
        walked.captured.cacheValue walked.captured.pathValue walked.captured.currentValue selected
      apply BlockContinues.branch (flag := true)
        (by simpa only [selected, Option.isSome_some] using tested) rfl
      rw [heap.set_old base walked.path walked.positions walked.garbage cacheAddress
        (.dict (store.encode finalEntries)) (store.found_inside found)] at executed
      simpa only [heap.work, ite_true] using executed
  | cycle index missing selected =>
      have complete := walked.cycleComplete missing
      have chosen := walked.recorded.sound walked.current index selected
      have inside : index < walked.path.length := (List.getElem?_eq_some_iff.mp chosen).1
      obtain ⟨correct, returned⟩ := memo.cycle_correct start walked.current index walked.path
        entries valid walked.visited complete walked.uncached chosen
      let locals := branches.cycleFinalLocals walked.locals walked.path index
      let finalEntries := branches.finishedEntries entries walked.path index
      let slices := [Mutable.Object.list ((walked.path.drop index).map number),
        .list ((walked.path.take index).map number)]
      let extra := Mutable.Object.list (walked.path.map number) ::
        .dict (store.encode walked.positions) :: (walked.garbage ++ slices)
      let finished : Written start cacheAddress :=
        ⟨finalEntries, extra, locals, correct, returned,
          by simpa [locals, branches.cycleFinalLocals, branches.prepared,
            write.tailLocals_frame, write.cycleLocals_frame, setLocal]
              using walked.captured.startValue,
          by simpa [locals, branches.cycleFinalLocals, branches.prepared,
            write.tailLocals_frame, write.cycleLocals_frame, setLocal]
              using walked.captured.cacheValue⟩
      refine ⟨finished, ?_⟩
      have executed := branches.cycle_result (globals := globals)
        (heap.work base walked.path walked.positions walked.garbage) walked.locals
        cacheAddress base.length (base.length + 1) walked.current index entries walked.positions
        walked.path inside.le
        (heap.old_found base walked.path walked.positions walked.garbage cacheAddress _ found)
        (heap.path_found base walked.path walked.positions walked.garbage)
        (heap.positions_found base walked.path walked.positions walked.garbage)
        walked.captured.cacheValue walked.captured.pathValue walked.captured.positionsValue
        walked.captured.currentValue selected
      apply BlockContinues.branch (flag := false)
        (by simpa only [missing, Option.isSome_none] using tested) rfl
      change BlockContinues program walked.locals write.cycleCode
        (heap.work base walked.path walked.positions walked.garbage) locals
        ((heap.work base walked.path walked.positions walked.garbage ++ slices).set
          cacheAddress (.dict (store.encode finalEntries))) globals at executed
      rw [heap.work_append, heap.set_old base walked.path walked.positions
        (walked.garbage ++ slices) cacheAddress _ (store.found_inside found)] at executed
      simpa only [heap.work, Bool.false_eq_true, ite_false] using executed

theorem body_source :
    executableBody ((findFunctionIn program.body "chain_length").getD ([], [])).2 =
      ((executableBody ((findFunctionIn program.body "chain_length").getD ([], [])).2).take 4) ++
        write.code ++ [.return (.subscript (.name "lengths") (.name "start"))] := rfl

/-- Every natural start extends the same dictionary and returns its distinct orbit length. -/
theorem result (start cacheAddress address : Nat) (base : Mutable.Heap)
    (entries : store.Entries) (valid : memo.Valid entries)
    (foundCache : store.Found base cacheAddress entries)
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (foundTable : base[address]? = some (.list factorials.values)) :
    ∃ finalEntries extra, memo.Valid finalEntries ∧
      store.lookup finalEntries start = some (chainLen start) ∧
      ReturnsValue program "chain_length" [number start, .ref cacheAddress] base
        (number (chainLen start))
        (base.set cacheAddress (.dict (store.encode finalEntries)) ++ extra) globals := by
  let args := ((findFunctionIn program.body "chain_length").getD ([], [])).1
  let source := ((findFunctionIn program.body "chain_length").getD ([], [])).2
  let entry := functionEnv args source [number start, .ref cacheAddress] globals
  have getStart : entry "start" = some (number start) := by cases globals <;> rfl
  have getCache : entry "lengths" = some (.ref cacheAddress) := by cases globals <;> rfl
  obtain ⟨walked, walkedSource⟩ := walk.forward_result (globals := globals)
    start cacheAddress address base entries entry getStart getCache foundCache getTable foundTable
  obtain ⟨finished, written⟩ := writeback (globals := globals)
    base start cacheAddress entries valid foundCache walked
  refine ⟨finished.entries, finished.extra, finished.valid, finished.selected, ?_⟩
  apply BlockReturns.function rfl rfl
  change BlockReturns program entry
    (executableBody ((findFunctionIn program.body "chain_length").getD ([], [])).2) base
    (number (chainLen start))
    (base.set cacheAddress (.dict (store.encode finished.entries)) ++ finished.extra) globals
  rw [body_source]
  have returned : BlockReturns program finished.locals
      [.return (.subscript (.name "lengths") (.name "start"))]
      (base.set cacheAddress (.dict (store.encode finished.entries)) ++ finished.extra)
      (number (chainLen start))
      (base.set cacheAddress (.dict (store.encode finished.entries)) ++ finished.extra) globals :=
    BlockReturns.return_value (store.lookup_evaluates (globals := globals)
      (base.set cacheAddress (.dict (store.encode finished.entries)) ++ finished.extra)
      finished.locals "lengths" "start" cacheAddress start (chainLen start) finished.entries
      (store.found_append (store.found_set foundCache finished.entries) finished.extra)
      finished.cacheValue finished.startValue finished.selected)
  intro anchors stack
  exact ((walkedSource.append written)
    [.return (.subscript (.name "lengths") (.name "start"))] anchors stack).trans
    (returned anchors stack)

end solvers.«74».chain_length
