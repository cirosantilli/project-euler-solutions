import solvers.«92.py.lean.d».TableCalls

namespace solvers.«92».rows
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := tableCalls.program
abbrev number := table.number

abbrev limit : Expr := .binOp
  (.ifExp (.name "tight") (.name "ceiling") (.constant (.int 9))) .add (.constant (.int 1))
abbrev loop : Stmt := .forIn (.name "digit") (.call (.name "range") [limit]) tableCalls.body []

theorem source :
    (match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[2]? with
      | some (Stmt.forIn _ _ prefixBody _) => match prefixBody[2]? with
          | some (Stmt.forDictionaryItems _ _ rowBody _) => rowBody
          | _ => []
      | _ => []) = [loop] := rfl

structure Ready (locals : HeapCalls.Env) (address : Nat) (state : Prefix.State) (ways ceiling : Nat) : Prop where
  dictionary : locals "following" = some (.ref address)
  total : locals "total" = some (number state.1)
  tight : locals "tight" = some (.scalar (.bool state.2))
  ways : locals "ways" = some (number ways)
  ceiling : locals "ceiling" = some (number ceiling)

/-- Only the digit and its freshly allocated key are rebound by the inner loop. -/
def LocalsFrame (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "digit" → name ≠ "key" → after name = before name

theorem Ready.frame {before after : HeapCalls.Env} {address : Nat} {state : Prefix.State}
    {ways ceiling : Nat} (ready : Ready before address state ways ceiling)
    (frame : LocalsFrame before after) : Ready after address state ways ceiling := by
  rcases ready with ⟨dict, total, tight, ways, ceiling⟩
  constructor <;> rw [frame _ (by decide) (by decide)] <;> assumption

/-- Existing objects other than the destination dictionary retain their identity and contents. -/
def HeapFrame (before after : Mutable.Heap) (address : Nat) : Prop :=
  ∀ other object, other ≠ address → before[other]? = some object → after[other]? = some object

theorem HeapFrame.step (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (extra : Mutable.Heap) : HeapFrame heap ((heap ++ extra).set address object) address := by
  intro other old different found
  rw [List.getElem?_set_ne (Ne.symm different)]
  exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found

theorem limit_result (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (address ways ceiling : Nat) (state : Prefix.State) (ready : Ready locals address state ways ceiling) :
    Evaluates program locals limit heap (number (Prefix.digits ceiling state.2)) heap globals := by
  apply Evaluates.binary (av := number (if state.2 then ceiling else 9)) (bv := number 1)
    (middle := heap) (afterRight := heap)
  · apply Evaluates.conditional (flag := state.2) (Evaluates.lookup ready.tight) rfl
    cases tight : state.2
    · exact Evaluates.constant
    · exact Evaluates.lookup ready.ceiling
  · exact Evaluates.constant
  · simp [Mutable.binary, scalarBinary, number, Prefix.digits]

def contributions (state : Prefix.State) (ways ceiling first count : Nat) : Prefix.Table :=
  (List.range' first count).map fun digit => (Prefix.advance ceiling state digit, ways)

theorem scan (count first : Nat) (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (address ways ceiling : Nat) (state : Prefix.State) (entries : table.Entries)
    (ready : Ready locals address state ways ceiling) (found : table.Found heap address entries)
    (keys : table.Keys heap entries) :
    ∃ result finalHeap finalLocals,
      BlockContinues program locals
        [.rangeStepLoop "digit" first (first + count) 1 tableCalls.body []] heap finalLocals finalHeap globals ∧
      table.Found finalHeap address result ∧ table.Keys finalHeap result ∧
      table.view result = Prefix.collect (table.view entries) (contributions state ways ceiling first count) ∧
      HeapFrame heap finalHeap address ∧ LocalsFrame locals finalLocals := by
  induction count generalizing first heap locals entries with
  | zero =>
      refine ⟨entries, heap, locals, ?_, found, keys, ?_, ?_, ?_⟩
      · apply BlockContinues.range_done _ .nil
        simp [Calls.rangeContinues]
      · simp [contributions, Prefix.collect]
      · intro other object different present; exact present
      · intro name digit key; rfl
  | succ count ih =>
      let state' := Prefix.advance ceiling state first
      let entries' := table.add entries state' heap.length ways
      let allocated := heap ++ [table.tuple state']
      let heap' := allocated.set address (.keyDict (table.encode entries'))
      let digitLocals := setLocal locals "digit" (number first)
      let locals' := setLocal digitLocals "key" (.ref heap.length)
      have localsFrame : LocalsFrame locals locals' := by
        intro name notDigit notKey
        simp [locals', digitLocals, setLocal, notDigit, notKey]
      have oldFound : table.Found allocated address entries := found.append _
      have nextFound : table.Found heap' address entries' := oldFound.set entries'
      have keyFound : allocated[heap.length]? = some (table.tuple state') := by simp [allocated]
      have nextKeys : table.Keys heap' entries' := by
        exact ((keys.append _).insert state' heap.length _ keyFound).set oldFound entries'
      have changed : HeapFrame heap heap' address := HeapFrame.step heap address _ _
      obtain ⟨result, finalHeap, finalLocals, ran, finalFound, finalKeys, view, heapFrame, localFrame⟩ :=
        ih (first + 1) heap' locals' entries' (ready.frame localsFrame) nextFound nextKeys
      refine ⟨result, finalHeap, finalLocals, ?_, finalFound, finalKeys, ?_, ?_, ?_⟩
      · apply BlockContinues.range_step
          (by simp [Calls.rangeContinues])
        apply BlockContinues.append
          (tableCalls.body_result heap digitLocals globals address ways first ceiling entries state found
            (by simpa [digitLocals, setLocal] using ready.dictionary)
            (by simpa [digitLocals, setLocal] using ready.ways)
            (by simpa [digitLocals, setLocal] using ready.total)
            (by simpa [digitLocals, setLocal] using ready.tight)
            (by simp [digitLocals, setLocal])
            (by simpa [digitLocals, setLocal] using ready.ceiling))
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          tableCalls.body, tableCalls.update, Int.add_assoc, Int.add_comm, Int.add_left_comm,
          locals', digitLocals, heap', allocated, entries', state'] using ran
      · rw [view]
        simp [contributions, List.range'_succ, Prefix.collect, entries', state', table.view_add]
      · intro other object different present
        exact heapFrame other object different (changed other object different present)
      · intro name notDigit notKey
        exact (localFrame name notDigit notKey).trans (localsFrame name notDigit notKey)

/-- The entire Python range loop realizes one weighted digit-prefix expansion row. -/
theorem result (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (address ways ceiling : Nat) (state : Prefix.State) (entries : table.Entries)
    (ready : Ready locals address state ways ceiling) (found : table.Found heap address entries)
    (keys : table.Keys heap entries) :
    ∃ result finalHeap finalLocals,
      BlockContinues program locals [loop] heap finalLocals finalHeap globals ∧
      table.Found finalHeap address result ∧ table.Keys finalHeap result ∧
      table.view result = Prefix.collect (table.view entries) (Prefix.row ceiling (state, ways)) ∧
      HeapFrame heap finalHeap address ∧ LocalsFrame locals finalLocals := by
  obtain ⟨result, finalHeap, finalLocals, ran, finalFound, finalKeys, view, heapFrame, localFrame⟩ :=
    scan (Prefix.digits ceiling state.2) 0 heap locals globals address ways ceiling state entries ready found keys
  refine ⟨result, finalHeap, finalLocals, ?_, finalFound, finalKeys, ?_, heapFrame, localFrame⟩
  · apply BlockContinues.for_range
      (.cons (limit_result heap locals globals address ways ceiling state ready) .nil)
      (by rfl)
    simpa using ran
  · simpa [contributions, Prefix.row, List.range_eq_range'] using view

end solvers.«92».rows
