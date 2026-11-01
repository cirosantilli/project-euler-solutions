import solvers.«92.py.lean.d».Rows
import lang.py.FormalSW.HeapMemoCallRules

namespace solvers.«92».initialization
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module

def expression : Expr :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[1]? with
  | some (Stmt.assign _ value) => value
  | _ => .constant .none

def keyExpression : Expr :=
  match expression with
  | .dict [(key, _)] => key
  | _ => .constant .none

def identity : String := (HeapMemoCalls.pooledIdentity keyExpression).getD ""
def poolKey : HeapMemoCalls.Key := HeapMemoCalls.literalKey identity

def entries (keyAddress : Nat) : table.Entries := [((0, true), keyAddress, 1)]

theorem source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[1]? =
      some (.assign [.name "counts"] expression) := rfl

theorem expression_eq : expression = .dict [(keyExpression, .constant (.int 1))] := rfl

theorem key_expression_eq : keyExpression = HeapMemoCalls.pooledTuple identity
    [.constant (.int 0), .constant (.bool true)] := rfl

def KeyFound (heap : Mutable.Heap) (value : Mutable.Value) : Prop :=
  ∃ address, value = .ref address ∧ heap[address]? = some (table.tuple (0, true))

def spec (heap : Mutable.Heap) (query : HeapMemoCalls.Key) (value : Mutable.Value) : Prop :=
  query = poolKey ∧ KeyFound heap value

theorem key_found_append (base extra : Mutable.Heap) (value : Mutable.Value) (found : KeyFound base value) :
    KeyFound (base ++ extra) value := by
  obtain ⟨address, rfl, found⟩ := found
  exact ⟨address, rfl, (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found⟩

theorem spec_preserves_append (base extra : Mutable.Heap) :
    HeapMemoCalls.Cache.Preserves spec base (base ++ extra) := by
  intro cache valid query value cached
  obtain ⟨same, found⟩ := valid query value cached
  exact ⟨same, key_found_append base extra value found⟩

/-- Dictionary updates cannot overwrite an immutable key tuple at the same address. -/
theorem spec_preserves_frame (before after : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : before[address]? = some object) (different : object ≠ table.tuple (0, true))
    (frame : rows.HeapFrame before after address) : HeapMemoCalls.Cache.Preserves spec before after := by
  intro cache valid query value cached
  obtain ⟨same, keyAddress, rfl, keyFound⟩ := valid query value cached
  have separate : keyAddress ≠ address := by
    intro equal
    subst keyAddress
    exact different (Option.some.inj (found.symm.trans keyFound))
  exact ⟨same, keyAddress, rfl, frame keyAddress _ separate keyFound⟩

def KeyPost (base : Mutable.Heap) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ extra, after = base ++ extra ∧ KeyFound after value

/-- Both pool hits and first-use allocation retain the actual source tuple's identity. -/
theorem key_result (base : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel program spec locals keyExpression base (KeyPost base) globals := by
  rw [key_expression_eq]
  apply HeapMemoCalls.EvaluatesRel.pooled_tuple
  · have allocated := HeapMemoCalls.Evaluates.tuple (module := program) (spec := spec)
      (env := locals) (heap := base) (globals := globals)
      (.cons (HeapMemoCalls.Evaluates.constant (value := .int 0))
        (.cons (HeapMemoCalls.Evaluates.constant (value := .bool true)) .nil))
      (spec_preserves_append base [table.tuple (0, true)]) (by simp)
    apply allocated.rel.consequence
    rintro value after ⟨rfl, rfl⟩
    have found : KeyFound (base ++ [table.tuple (0, true)]) (.ref base.length) :=
      ⟨base.length, rfl, by simp⟩
    exact ⟨⟨[table.tuple (0, true)], rfl, found⟩, rfl, found⟩
  · intro value cached
    exact ⟨[], by simp, cached.2⟩

def Post (base : Mutable.Heap) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ keyAddress address extra, value = .ref address ∧ after = base ++ extra ∧
    table.Found after address (entries keyAddress) ∧ table.Keys after (entries keyAddress)

/-- The source literal creates the exact initial distribution and retains its pooled key. -/
theorem result (base : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel program spec locals expression base (Post base) globals := by
  rw [expression_eq]
  intro anchors stack
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.dictionary [(keyExpression, .constant (.int 1))] [] locals, stack, base⟩)
    (by rfl) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.expr keyExpression locals,
      .dictionaryKey (.constant (.int 1)) [] [] locals :: stack, base⟩) (by trivial) rfl
  apply (key_result base locals globals anchors _).bind
  rintro final ⟨value, middle, rfl, extra, same, address, rfl, found⟩
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.expr (.constant (.int 1)) locals,
      .arguments .dictionarySingleton [.ref address] [] locals :: stack, middle⟩) (by trivial) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.value (.scalar (.int 1)),
      .arguments .dictionarySingleton [.ref address] [] locals :: stack, middle⟩) (by rfl) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.operation .dictionarySingleton [.ref address, .scalar (.int 1)], stack, middle⟩)
    (by trivial) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.value (.ref middle.length), stack,
      middle ++ [.keyDict (table.encode (entries address))]⟩)
    (preserved := spec_preserves_append _ _) (by trivial)
  · simp [HeapCalls.step, primitive, HeapCalls.result, HeapCalls.run,
      DictionaryDisplay.singleton_result middle (.ref address) (.scalar (.int 1)) _
        (table.hash middle address (0, true) found), table.encode, entries]
  · apply HeapMemoCalls.RunsRel.refl
    refine ⟨.ref middle.length, middle ++ [.keyDict (table.encode (entries address))], rfl,
      address, middle.length, extra ++ [.keyDict (table.encode (entries address))], rfl, ?_, ?_, ?_⟩
    · simp [same, List.append_assoc]
    · exact Or.inl (by simp)
    · intro entry member
      simp only [entries, List.mem_singleton] at member
      subst entry
      exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found

theorem exact (keyAddress : Nat) : Prefix.Exact 0 (table.view (entries keyAddress)) :=
  Prefix.initial_exact

end solvers.«92».initialization
