import «.formalsw».solvers.«54.py.ast»
import lang.py.FormalSW.DictionaryCalls

/-! Source proofs for poker-card parsing and hand evaluation. -/
set_option Elab.async false
namespace solvers.«54»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P54Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private abbrev character (c : Char) : Mutable.Value := .scalar (.string (String.singleton c))

namespace parse_card
private theorem first_item (token : String) (rank suit : Char) (rest : List Char)
    (digits : token.toList = rank :: suit :: rest) (heap : Mutable.Heap) :
    Mutable.getItem (.scalar (.string token)) (number 0) heap = some (.ok (character rank), heap) := by
  have inside : ¬ (rest.length : Int)+1+1 ≤ 0 := by omega
  simp [Mutable.getItem, Strings.get, FormalSW.Py.Builtin.Sequence.get?,
    FormalSW.Py.Builtin.Sequence.normalizeIndex, digits, character, inside]

private theorem second_item (token : String) (rank suit : Char) (rest : List Char)
    (digits : token.toList = rank :: suit :: rest) (heap : Mutable.Heap) :
    Mutable.getItem (.scalar (.string token)) (number 1) heap = some (.ok (character suit), heap) := by
  have inside : ¬ (rest.length : Int)+1+1 ≤ 1 := by omega
  simp [Mutable.getItem, Strings.get, FormalSW.Py.Builtin.Sequence.get?,
    FormalSW.Py.Builtin.Sequence.normalizeIndex, digits, character, inside]
/-- The parser reads the supplied rank dictionary and allocates only its returned pair. -/
theorem result (token : String) (rank suit : Char) (rest : List Char) (digits : token.toList = rank :: suit :: rest)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref address)) (found : heap[address]? = some (.dict entries))
    (value : Mutable.Value) (present : ScalarDict.lookup entries (.string (String.singleton rank)) = some value) :
    ReturnsValue program "parse_card" [.scalar (.string token)] heap (.ref heap.length)
      (heap ++ [.tuple [value, character suit]]) globals := by
  let entry := functionEnv ((findFunctionIn program.body "parse_card").getD ([], [])).1
    ((findFunctionIn program.body "parse_card").getD ([], [])).2 [.scalar (.string token)] globals
  let ranked := setLocal entry "v" value
  let ready := setLocal ranked "s" (character suit)
  have getToken : entry "token" = some (.scalar (.string token)) := by cases globals <;> rfl
  have table : entry "VALUE_MAP" = some (.ref address) := by
    cases globals with
    | nil => simp at getTable
    | cons pair tail => exact getTable
  have first : Evaluates program entry (.subscript (.name "token") (.constant (.int 0))) heap (character rank) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap)
      (by intro a b c impossible; cases impossible) (Evaluates.lookup getToken) Evaluates.constant
    exact first_item token rank suit rest digits heap
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := value) (middle := heap)
  · apply Evaluates.subscript (middle := heap) (afterIndex := heap)
      (by intro a b c impossible; cases impossible) (Evaluates.lookup table) first
    exact Mutable.getItem_dictionary heap address entries found (.string (String.singleton rank)) present
  · apply BlockReturns.assign (value := character suit) (middle := heap)
    · apply Evaluates.subscript (middle := heap) (afterIndex := heap)
        (by intro a b c impossible; cases impossible) (Evaluates.lookup (by simpa [ranked, setLocal] using getToken)) Evaluates.constant
      exact second_item token rank suit rest digits heap
    · apply BlockReturns.return_value
      exact Evaluates.tuple (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil)) (by simp)
private abbrev context (address : Nat) : Mutable.Globals := [("VALUE_MAP", .ref address)]
private abbrev inputEnv (token : String) (address : Nat) : HeapCalls.Env :=
  fun name => if name = "token" then some (.scalar (.string token)) else (context address).lookup name
private abbrev longEnough (heap : Mutable.Heap) (token : String) (address : Nat) :=
  HeapCalls.Condition program (inputEnv token address) Solvers.P54Ast.requires_parse_card_0 heap (context address)
private abbrev knownRank (heap : Mutable.Heap) (token : String) (address : Nat) :=
  HeapCalls.Condition program (inputEnv token address) Solvers.P54Ast.requires_parse_card_1 heap (context address)

private theorem length_test (heap : Mutable.Heap) (token : String) (address : Nat) :
    Evaluates program (inputEnv token address) Solvers.P54Ast.requires_parse_card_0
      heap (flag (decide (2 ≤ token.toList.length))) heap (context address) := by
  apply Evaluates.compare (av := number token.toList.length) (bv := number 2) (middle := heap) (afterRight := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
    simp [primitive, Mutable.length, number]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega

private theorem token_parts (heap : Mutable.Heap) (token : String) (address : Nat) (enough : longEnough heap token address) :
    ∃ rank suit rest, token.toList = rank :: suit :: rest := by
  have size : 2 ≤ token.toList.length := by simpa [flag] using (length_test heap token address).condition_iff.mp enough
  cases digits : token.toList with
  | nil => simp [digits] at size
  | cons rank tail =>
      cases tail with
      | nil => simp [digits] at size
      | cons suit rest => exact ⟨rank, suit, rest, rfl⟩

private theorem membership_test (heap : Mutable.Heap) (token : String) (address : Nat)
    (rank suit : Char) (rest : List Char) (digits : token.toList = rank :: suit :: rest)
    (entries : List (Constant × Mutable.Value)) (found : heap[address]? = some (.dict entries)) :
    Evaluates program (inputEnv token address) Solvers.P54Ast.requires_parse_card_1 heap
      (flag (ScalarDict.lookup entries (.string (String.singleton rank))).isSome) heap (context address) := by
  apply Evaluates.membership (middle := heap) (needle := character rank) (source := .ref address)
  · refine .cons ?_ (.cons (Evaluates.lookup rfl) .nil)
    exact Evaluates.subscript (by intro a b c impossible; cases impossible) (Evaluates.lookup rfl) Evaluates.constant
      (first_item token rank suit rest digits heap)
  · exact membership_dictionary (.string (String.singleton rank)) false found

theorem «Returns» (heap : Mutable.Heap) (token : String) (address : Nat)
    (_valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "dict[str,int]")
    (enough : longEnough heap token address) (known : knownRank heap token address) :
    (HeapCalls.call program "parse_card" [.scalar (.string token)] heap (context address)).Returns := by
  obtain ⟨rank, suit, rest, digits⟩ := token_parts heap token address enough
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have present : (ScalarDict.lookup entries (.string (String.singleton rank))).isSome = true := by
    simpa [flag] using (membership_test heap token address rank suit rest digits entries found).condition_iff.mp known
  cases selected : ScalarDict.lookup entries (.string (String.singleton rank)) with
  | none => simp [selected] at present
  | some value => exact (result token rank suit rest digits heap (context address) address entries rfl found value selected).returns

theorem «Supported» (heap : Mutable.Heap) (token : String) (address : Nat)
    (valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "dict[str,int]")
    (enough : longEnough heap token address) (known : knownRank heap token address) :
    (HeapCalls.call program "parse_card" [.scalar (.string token)] heap (context address)).Supported :=
  («Returns» heap token address valid typed enough known).supported

end parse_card
theorem requires_parse_card_0_supported (heap : Mutable.Heap) (token : String) (address : Nat)
    (_valid : heap.Valid) (_typed : Mutable.Value.HasType heap (.ref address) "dict[str,int]") :
    (HeapCalls.evaluate program (parse_card.inputEnv token address) Solvers.P54Ast.requires_parse_card_0 heap (parse_card.context address)).Supported :=
  Execution.supported_of_finishes (parse_card.length_test heap token address).finishes

theorem requires_parse_card_1_supported (heap : Mutable.Heap) (token : String) (address : Nat)
    (_valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "dict[str,int]")
    (enough : parse_card.longEnough heap token address) :
    (HeapCalls.evaluate program (parse_card.inputEnv token address) Solvers.P54Ast.requires_parse_card_1 heap (parse_card.context address)).Supported := by
  obtain ⟨rank, suit, rest, digits⟩ := parse_card.token_parts heap token address enough
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  exact Execution.supported_of_finishes (parse_card.membership_test heap token address rank suit rest digits entries found).finishes

end solvers.«54»
