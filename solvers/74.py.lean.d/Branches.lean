import solvers.«74.py.lean.d».Write

/-! Both writeback branches execute with fresh slices and a shared mutable cache. -/
namespace solvers.«74».branches
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P74Ast.module
abbrev number := factorials.number
variable {globals : Mutable.Globals}

theorem suffix_slice (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (address index : Nat) (values : List Nat)
    (found : heap[address]? = some (.list (values.map number)))
    (getPath : locals "path" = some (.ref address))
    (getIndex : locals "cycle_start" = some (number index)) :
    Evaluates program locals write.cycleIterable heap (.ref heap.length)
      (heap ++ [.list ((values.drop index).map number)]) globals := by
  apply Evaluates.slice (.cons (Evaluates.lookup getPath)
    (.cons (Evaluates.lookup getIndex) (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
  simpa only [List.map_drop] using
    Mutable.getSlice_list_suffix heap address (values.map number) index found

theorem prefix_slice (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (address index : Nat) (values : List Nat) (inside : index ≤ values.length)
    (found : heap[address]? = some (.list (values.map number)))
    (getPath : locals "path" = some (.ref address))
    (getIndex : locals "cycle_start" = some (number index)) :
    Evaluates program locals write.prefixIterable heap (.ref heap.length)
      (heap ++ [.list ((values.take index).map number)]) globals := by
  apply Evaluates.slice (.cons (Evaluates.lookup getPath)
    (.cons Evaluates.constant (.cons (Evaluates.lookup getIndex) (.cons Evaluates.constant .nil))))
  simpa only [FormalSW.Py.Builtin.Slice.prefixLength_nonnegative, List.length_map,
    Nat.min_eq_left inside, List.map_take] using
      Mutable.getSlice_list_prefix heap address (values.map number) index found

theorem cached_result (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress pathAddress current remaining : Nat) (entries : store.Entries) (path : List Nat)
    (foundCache : store.Found heap cacheAddress entries)
    (foundPath : heap[pathAddress]? = some (.list (path.map number)))
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (getPath : locals "path" = some (.ref pathAddress))
    (getCurrent : locals "current" = some (number current))
    (selected : store.lookup entries current = some remaining) :
    BlockContinues program locals write.cachedCode heap
      (write.tailLocals (setLocal locals "remaining" (number remaining)) remaining path.reverse)
      (heap.set cacheAddress (.dict (store.encode (write.tailFill entries remaining path.reverse))))
      globals := by
  have assigned := BlockContinues.assign (name := "remaining")
    (store.lookup_evaluates (globals := globals) heap locals "lengths" "current"
      cacheAddress current remaining entries foundCache getCache getCurrent selected)
  have filled := write.reverse_result (globals := globals) heap heap
    (setLocal locals "remaining" (number remaining)) cacheAddress pathAddress remaining entries
    path (.name "path") (Evaluates.lookup (by simpa [setLocal] using getPath))
    foundCache foundPath (by simpa [setLocal] using getCache) (by simp [setLocal])
  exact assigned.append filled

abbrev cycleEntries (entries : store.Entries) (path : List Nat) (index : Nat) : store.Entries :=
  write.cycleFill entries (path.length - index) (path.drop index)

abbrev finishedEntries (entries : store.Entries) (path : List Nat) (index : Nat) : store.Entries :=
  write.tailFill (cycleEntries entries path index) (path.length - index) (path.take index).reverse

abbrev prepared (locals : HeapCalls.Env) (path : List Nat) (index : Nat) : HeapCalls.Env :=
  setLocal (setLocal locals "cycle_start" (number index))
    "cycle_length" (number (path.length - index))

abbrev cycleFinalLocals (locals : HeapCalls.Env) (path : List Nat) (index : Nat) : HeapCalls.Env :=
  write.tailLocals
    (setLocal (write.cycleLocals (prepared locals path index) (path.drop index))
      "remaining" (number (path.length - index)))
    (path.length - index) (path.take index).reverse

/-- The suffix is cached as a complete batch before the reverse prefix writes begin. -/
theorem cycle_result (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress pathAddress positionsAddress current index : Nat)
    (entries pos : store.Entries) (path : List Nat) (inside : index ≤ path.length)
    (foundCache : store.Found heap cacheAddress entries)
    (foundPath : heap[pathAddress]? = some (.list (path.map number)))
    (foundPositions : store.Found heap positionsAddress pos)
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (getPath : locals "path" = some (.ref pathAddress))
    (getPositions : locals "positions" = some (.ref positionsAddress))
    (getCurrent : locals "current" = some (number current))
    (selected : store.lookup pos current = some index) :
    BlockContinues program locals write.cycleCode heap (cycleFinalLocals locals path index)
      ((heap ++ [Mutable.Object.list ((path.drop index).map number),
        Mutable.Object.list ((path.take index).map number)]).set
        cacheAddress (.dict (store.encode (finishedEntries entries path index)))) globals := by
  let length := path.length - index
  let staged := prepared locals path index
  let suffix := Mutable.Object.list ((path.drop index).map number)
  let front := Mutable.Object.list ((path.take index).map number)
  let cycleHeap := (heap ++ [suffix]).set cacheAddress
    (.dict (store.encode (cycleEntries entries path index)))
  let cycleEnv := write.cycleLocals staged (path.drop index)
  let tailEnv := setLocal cycleEnv "remaining" (number length)
  have assigned := BlockContinues.assign (name := "cycle_start")
    (store.lookup_evaluates (globals := globals) heap locals "positions" "current"
      positionsAddress current index pos foundPositions getPositions getCurrent selected)
  have lengthValue : Evaluates program (setLocal locals "cycle_start" (number index))
      (.binOp (.builtinCall .len [.name "path"]) .sub (.name "cycle_start")) heap
      (number length) heap globals := by
    apply Evaluates.binary (av := number path.length) (bv := number index)
      (middle := heap) (afterRight := heap)
    · apply Evaluates.builtin (.cons (Evaluates.lookup (by simpa [setLocal] using getPath)) .nil)
      simp [primitive, Mutable.length, foundPath, number, factorials.number]
    · exact Evaluates.lookup (by simp [setLocal])
    · simp [Mutable.binary, scalarBinary, number, factorials.number, length,
        Int.ofNat_sub inside]
  have stagedCache : staged "lengths" = some (.ref cacheAddress) := by
    simpa [staged, prepared, setLocal] using getCache
  have stagedPath : staged "path" = some (.ref pathAddress) := by
    simpa [staged, prepared, setLocal] using getPath
  have stagedIndex : staged "cycle_start" = some (number index) := by
    simp [staged, prepared, setLocal]
  have stagedLength : staged "cycle_length" = some (number length) := by
    simp [staged, prepared, setLocal, length]
  have sliced := suffix_slice (globals := globals) heap staged pathAddress index path foundPath
    stagedPath stagedIndex
  have cycleWritten := write.cycle_result (globals := globals) heap (heap ++ [suffix]) staged
    cacheAddress heap.length length entries (path.drop index) write.cycleIterable
    (by intro expressions; simp [write.cycleIterable]) sliced
    (store.found_append foundCache [suffix]) (by simp [suffix]) stagedCache stagedLength
  have cacheInside := store.found_inside foundCache
  have different : cacheAddress ≠ pathAddress := by
    intro same
    subst pathAddress
    rw [foundCache] at foundPath
    cases foundPath
  have retainedPath : cycleHeap[pathAddress]? = some (.list (path.map number)) := by
    change ((heap ++ [suffix]).set cacheAddress _)[pathAddress]? = _
    rw [List.getElem?_set_ne different]
    exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp foundPath).1).trans foundPath
  have tailCache : tailEnv "lengths" = some (.ref cacheAddress) := by
    simpa only [tailEnv, setLocal, reduceCtorEq, if_false,
      cycleEnv, write.cycleLocals_frame staged _ "lengths" (by decide)] using stagedCache
  have tailPath : tailEnv "path" = some (.ref pathAddress) := by
    simpa only [tailEnv, setLocal, reduceCtorEq, if_false,
      cycleEnv, write.cycleLocals_frame staged _ "path" (by decide)] using stagedPath
  have tailIndex : tailEnv "cycle_start" = some (number index) := by
    simpa [tailEnv, setLocal, cycleEnv, write.cycleLocals_frame] using stagedIndex
  have frontSliced := prefix_slice (globals := globals) cycleHeap tailEnv pathAddress index path
    inside retainedPath tailPath tailIndex
  have tailWritten := write.reverse_result (globals := globals) cycleHeap (cycleHeap ++ [front])
    tailEnv cacheAddress cycleHeap.length length (cycleEntries entries path index) (path.take index)
    write.prefixIterable frontSliced
    (store.found_append (store.found_set (store.found_append foundCache [suffix]) _) [front])
    (by simp [front]) tailCache (by simp [tailEnv, setLocal])
  have remainingAssigned : BlockContinues program cycleEnv
      [.assign [.name "remaining"] (.name "cycle_length")] cycleHeap tailEnv cycleHeap globals :=
    BlockContinues.assign (Evaluates.lookup (by
      simpa only [cycleEnv, write.cycleLocals_frame staged _ "cycle_length" (by decide)] using
        stagedLength))
  have combined := assigned.append
    ((BlockContinues.assign (name := "cycle_length") lengthValue).append
      (cycleWritten.append (remainingAssigned.append tailWritten)))
  have normalized :
      (cycleHeap ++ [front]).set cacheAddress
          (.dict (store.encode (finishedEntries entries path index))) =
        (heap ++ [suffix, front]).set cacheAddress
          (.dict (store.encode (finishedEntries entries path index))) := by
    have bound : cacheAddress < (heap ++ [suffix]).length := by
      simp only [List.length_append, List.length_singleton]
      omega
    change (((heap ++ [suffix]).set cacheAddress _ ++ [front]).set cacheAddress _) = _
    rw [← List.set_append_left cacheAddress _ bound, List.set_set, List.append_assoc]
    rfl
  change BlockContinues program locals write.cycleCode heap
    (cycleFinalLocals locals path index)
    ((cycleHeap ++ [front]).set cacheAddress
      (.dict (store.encode (finishedEntries entries path index)))) globals at combined
  rw [normalized] at combined
  exact combined

end solvers.«74».branches
