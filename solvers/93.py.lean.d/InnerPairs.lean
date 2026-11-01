import solvers.«93.py.lean.d».PairBody
import ProjectEulerSolutions.Lib.RationalBranches

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions
set_option Elab.async false

/-- Scan every later operand position while retaining the memo and private collector invariants. -/
theorem inner_pairs (pairs : List Pair) (i : Nat) (canonical : ∀ pair ∈ pairs, Canonical pair)
    (globals : Mutable.Globals) (smaller : SmallerCalls pairs.length globals)
    (cell memo collector state : Nat) (values : List Mutable.Value)
    (count first : Nat) (ordered : i < first) (stop : first + count = pairs.length)
    (heap : Mutable.Heap) (entries : List Memo.Entry) (collected : List Heap.SetEntry) (env : HeapCalls.Env)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (stateFound : heap[state]? = some (.tuple values)) (rows : Heap.Rows heap values pairs)
    (original : Heap.SetAt heap collector collected) (uncached : collector ∉ entries.map Memo.Entry.result)
    (getCell : env cellName = some (.ref cell)) (getState : env "state" = some (.ref state))
    (getI : env "i" = some (.scalar (.int i))) (getResult : env "results" = some (.ref collector)) :
    ∃ after finalEnv final finalEntries,
      BlockContinues program env [.rangeStepLoop "j" first pairs.length 1 pairBody []] heap finalEnv after globals ∧
      Heap.SetAt after collector final ∧
      (∀ pair, pair ∈ final.map Prod.fst ↔ pair ∈ collected.map Prod.fst ∨
        ∃ j ∈ List.range' first count, PairExpected pairs i j pair) ∧
      MemoValid after memo finalEntries ∧ Memo.Origins entries heap.length finalEntries ∧
      collector ∉ finalEntries.map Memo.Entry.result ∧ PreservesOutside [memo, collector] heap after ∧
      (∀ name, name ≠ "j" → name ≠ "rest" → name ≠ "value" → finalEnv name = env name) := by
  induction count generalizing first heap entries collected env with
  | zero =>
    refine ⟨heap, env, collected, entries, ?_, original, ?_, valid, Memo.Origins.refl entries heap.length,
      uncached, PreservesOutside.refl _ heap, fun _ _ _ _ => rfl⟩
    · apply BlockContinues.range_done (rest := .nil)
      simp [Calls.rangeContinues]
      omega
    · intro pair
      simp
  | succ count ih =>
    have inside : first < pairs.length := by omega
    let bound := setLocal env "j" (.scalar (.int first))
    obtain ⟨middle, middleEnv, next, middleEntries, executed, nextSet, firstMembers, middleMemo, firstOrigins, absent, firstFrame, firstLocals⟩ :=
      pair_body heap cell memo collector state values pairs i first ordered inside canonical entries collected bound globals smaller
        cellFound valid stateFound rows original uncached
        (by simpa [bound, setLocal, cellName] using getCell) (by simpa [bound, setLocal] using getState)
        (by simpa [bound, setLocal] using getI) (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getResult)
    have nextCell := firstFrame.ordinary (Memo.encode entries) (collected.map Heap.encodeEntry)
      valid.stored original.keyEntries cell _ cellFound rfl rfl
    have nextState := firstFrame.ordinary (Memo.encode entries) (collected.map Heap.encodeEntry)
      valid.stored original.keyEntries state _ stateFound rfl rfl
    have nextRows := rows.preserved_outside firstFrame (Memo.encode entries) (collected.map Heap.encodeEntry) valid.stored original.keyEntries
    have nextGet (name : String) (notRest : name ≠ "rest") (notValue : name ≠ "value") (notJ : name ≠ "j") :
        middleEnv name = env name := by
      rw [firstLocals name notRest notValue]
      simp [bound, setLocal, notJ]
    obtain ⟨after, finalEnv, final, finalEntries, continued, finalSet, finalMembers, finalMemo, finalOrigins, finalAbsent, finalFrame, finalLocals⟩ :=
      ih (first + 1) (by omega) (by omega) middle middleEntries next middleEnv nextCell middleMemo nextState nextRows nextSet absent
        (by rw [nextGet cellName (by decide) (by decide) (by decide)]; exact getCell)
        (by rw [nextGet "state" (by decide) (by decide) (by decide)]; exact getState)
        (by rw [nextGet "i" (by decide) (by decide) (by decide)]; exact getI)
        (by rw [nextGet "results" (by decide) (by decide) (by decide)]; exact getResult)
    refine ⟨after, finalEnv, final, finalEntries, ?_, finalSet, ?_, finalMemo,
      firstOrigins.trans finalOrigins firstFrame.size, finalAbsent, firstFrame.trans finalFrame, ?_⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have control : Calls.loopControl pairBody = false := rfl
      simpa only [Calls.loopContinuation, control, Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using
        executed.append continued
    · intro pair
      rw [finalMembers, firstMembers, ← pairExpected_iff pairs i first ordered inside pair]
      simp only [List.range'_succ, List.mem_cons, or_and_right, exists_or, exists_eq_left]
      tauto
    · intro name notJ notRest notValue
      exact (finalLocals name notJ notRest notValue).trans (nextGet name notRest notValue notJ)

abbrev rowBody : List Stmt := [.forIn (.name "j")
  (.call (.name "range") [.binOp (.name "i") .add (.constant (.int 1)), .builtinCall .len [.name "state"]]) pairBody []]

theorem source_pair_loops : recBody[3]? = some
    (.forIn (.name "i") (.call (.name "range") [.builtinCall .len [.name "state"]]) rowBody []) := rfl

theorem inner_row (heap : Mutable.Heap) (cell memo collector state : Nat) (values : List Mutable.Value)
    (pairs : List Pair) (i : Nat) (inside : i < pairs.length) (canonical : ∀ pair ∈ pairs, Canonical pair)
    (entries : List Memo.Entry) (collected : List Heap.SetEntry) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (smaller : SmallerCalls pairs.length globals) (cellFound : heap[cell]? = some (.list [.ref memo]))
    (valid : MemoValid heap memo entries) (stateFound : heap[state]? = some (.tuple values)) (rows : Heap.Rows heap values pairs)
    (original : Heap.SetAt heap collector collected) (uncached : collector ∉ entries.map Memo.Entry.result)
    (getCell : env cellName = some (.ref cell)) (getState : env "state" = some (.ref state))
    (getI : env "i" = some (.scalar (.int i))) (getResult : env "results" = some (.ref collector)) :
    ∃ after finalEnv final finalEntries,
      BlockContinues program env rowBody heap finalEnv after globals ∧ Heap.SetAt after collector final ∧
      (∀ pair, pair ∈ final.map Prod.fst ↔ pair ∈ collected.map Prod.fst ∨ ∃ j, PairExpected pairs i j pair) ∧
      MemoValid after memo finalEntries ∧ Memo.Origins entries heap.length finalEntries ∧
      collector ∉ finalEntries.map Memo.Entry.result ∧ PreservesOutside [memo, collector] heap after ∧
      (∀ name, name ≠ "j" → name ≠ "rest" → name ≠ "value" → finalEnv name = env name) := by
  obtain ⟨after, finalEnv, final, finalEntries, executed, finalSet, members, finalMemo, origins, absent, frame, locals⟩ :=
    inner_pairs pairs i canonical globals smaller cell memo collector state values (pairs.length - (i + 1)) (i + 1)
      (by omega) (by omega) heap entries collected env cellFound valid stateFound rows original uncached getCell getState getI getResult
  have nextI : Evaluates program env (.binOp (.name "i") .add (.constant (.int 1))) heap (.scalar (.int (i + 1))) heap globals := by
    apply Evaluates.binary (Evaluates.lookup getI) Evaluates.constant
    simp [Mutable.binary, scalarBinary]
  have size : Evaluates program env (.builtinCall .len [.name "state"]) heap (.scalar (.int pairs.length)) heap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getState) .nil)
    simp [primitive, Mutable.length, stateFound, rows.length_eq]
  refine ⟨after, finalEnv, final, finalEntries, BlockContinues.for_range (.cons nextI (.cons size .nil)) rfl executed,
    finalSet, ?_, finalMemo, origins, absent, frame, locals⟩
  intro pair
  rw [members]
  apply or_congr Iff.rfl
  constructor
  · rintro ⟨j, _, expected⟩
    exact ⟨j, expected⟩
  · rintro ⟨j, expected⟩
    refine ⟨j, ?_, expected⟩
    obtain ⟨ordered, upper, _⟩ := expected
    rw [List.mem_range'_1]
    constructor <;> omega

end solvers.«93».all_results
