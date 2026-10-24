import «.formalsw».solvers.«54.py.ast»
import lang.py.FormalSW.DictionaryCalls
import lang.py.FormalSW.TupleComparisons

/-! Source proofs for poker-card parsing and hand evaluation. -/
set_option Elab.async false
namespace solvers.«54»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P54Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private abbrev character (c : Char) : Mutable.Value := .scalar (.string (String.singleton c))

namespace count_values

/-- Ordered integer counts retain the first position at which each key appeared. -/
def lookup : List (Int × Int) → Int → Int
  | [], _ => 0
  | (stored, value) :: rest, key => if stored = key then value else lookup rest key

def put : List (Int × Int) → Int → Int → List (Int × Int)
  | [], key, value => [(key, value)]
  | (stored, previous) :: rest, key, value =>
      if stored = key then (stored, value) :: rest else (stored, previous) :: put rest key value

def bump (table : List (Int × Int)) (key : Int) : List (Int × Int) := put table key (lookup table key + 1)

def counts (values : List Int) : List (Int × Int) := values.foldl bump []

def entries (table : List (Int × Int)) : List (Constant × Mutable.Value) :=
  table.map fun (key, value) => (.int key, number value)

private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev working (heap : Mutable.Heap) (table : List (Int × Int)) : Mutable.Heap := heap ++ [.dict (entries table)]

theorem lookup_put (table : List (Int × Int)) (key value query : Int) :
    lookup (put table key value) query = if query = key then value else lookup table query := by
  induction table with
  | nil => by_cases same : query = key <;> simp [put, lookup, same, Ne.symm]
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      by_cases updated : stored = key <;> by_cases selected : stored = query <;>
        by_cases same : query = key <;> simp_all [put, lookup]

theorem lookup_bump (table : List (Int × Int)) (key query : Int) :
    lookup (bump table key) query = lookup table query + if query = key then 1 else 0 := by
  rw [bump, lookup_put]
  by_cases same : query = key <;> simp [same]

private theorem lookup_fold (values : List Int) (table : List (Int × Int)) (query : Int) :
    lookup (values.foldl bump table) query = lookup table query + (values.count query : Int) := by
  induction values generalizing table with
  | nil => simp
  | cons value rest ih =>
      rw [List.foldl_cons, ih, lookup_bump]
      by_cases same : query = value <;> simp [same, Ne.symm] <;> omega

/-- The dictionary lookup, defaulting to zero, equals the exact number of occurrences. -/
theorem lookup_counts (values : List Int) (query : Int) :
    lookup (counts values) query = (values.count query : Int) := by
  simpa [counts, lookup] using lookup_fold values [] query

private theorem entries_put (table : List (Int × Int)) (key value : Int) :
    entries (put table key value) = ScalarDict.insert (entries table) (.int key) (number value) := by
  induction table with
  | nil => rfl
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      by_cases same : stored = key
      · simp [put, entries, ScalarDict.insert, scalarCompare, evalCmpOp, same]
      · simpa [put, entries, ScalarDict.insert, scalarCompare, evalCmpOp, same] using
          congrArg (List.cons (Constant.int stored, number previous)) ih

private theorem lookup_entries (table : List (Int × Int)) (key : Int) :
    (ScalarDict.lookup (entries table) (.int key)).getD (number 0) = number (lookup table key) := by
  induction table with
  | nil => rfl
  | cons pair rest ih =>
      obtain ⟨stored, value⟩ := pair
      by_cases same : stored = key
      · simp [entries, ScalarDict.lookup, lookup, scalarCompare, evalCmpOp, same]
      · simpa [entries, ScalarDict.lookup, lookup, scalarCompare, evalCmpOp, same] using ih

private def body : List Stmt :=
  [.assign [.subscript (.name "counts") (.name "value")]
    (.binOp (.call (.attribute (.name "counts") "get") [.name "value", .constant (.int 0)]) .add (.constant (.int 1)))]

private theorem input_found (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.list (numbers values))) (table : List (Int × Int)) :
    (working heap table)[address]? = some (.list (numbers values)) := by
  rw [working, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem increment (heap : Mutable.Heap) (table : List (Int × Int)) (value : Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getCounts : env "counts" = some (.ref heap.length)) (getValue : env "value" = some (number value)) :
    BlockContinues program env body (working heap table) env (working heap (bump table value)) globals := by
  apply BlockContinues.setItem (by intro a b c impossible; cases impossible)
    (afterRhs := working heap table) (afterTarget := working heap table)
    (value := number (lookup table value + 1))
  · apply Evaluates.binary (middle := working heap table) (afterRight := working heap table)
      (av := number (lookup table value)) (bv := number 1)
    · apply Evaluates.dictionary_get (entries := entries table)
        (by intro name equal; cases equal; simp [getCounts])
        (Evaluates.lookup getCounts) (by simp [working])
        (.cons (Evaluates.lookup getValue) (.cons Evaluates.constant .nil))
      simpa only [lookup_entries] using
        Mutable.dictionaryGet_scalar (working heap table) heap.length (entries table) (by simp [working]) (.int value) (number 0)
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, number]
  · exact .cons (Evaluates.lookup getCounts) (.cons (Evaluates.lookup getValue) .nil)
  · simpa [working, bump, entries_put, List.set_append_right] using
      Mutable.setItem_dictionary (working heap table) heap.length (entries table) (by simp [working])
        (.int value) (number (lookup table value + 1))

private theorem scan (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.list (numbers values))) (remaining before : List Int)
    (decomposition : values = before ++ remaining) (table : List (Int × Int))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getCounts : env "counts" = some (.ref heap.length)) :
    BlockReturns program env
      [.sequenceStepLoop "value" (.ref address) before.length body [], .return (.name "counts")]
      (working heap table) (.ref heap.length) (working heap (remaining.foldl bump table)) globals := by
  induction remaining generalizing before table env with
  | nil =>
      apply BlockReturns.sequence_done (items := numbers values)
      · simp [sequenceValue, Mutable.sequence, input_found heap address values found table]
      · simp [numbers, decomposition]
      · exact BlockReturns.return_value (Evaluates.lookup getCounts)
  | cons value rest ih =>
      let nextEnv := setLocal env "value" (number value)
      apply BlockReturns.sequence_step (items := numbers values) (value := number value)
      · simp [sequenceValue, Mutable.sequence, input_found heap address values found table]
      · simp [numbers, decomposition]
      · apply (increment heap table value nextEnv globals (by simpa [nextEnv, setLocal] using getCounts) rfl).returns
        change BlockReturns program nextEnv
          [.sequenceStepLoop "value" (.ref address) (before.length+1) body [], .return (.name "counts")]
          (working heap (bump table value)) (.ref heap.length) (working heap (rest.foldl bump (bump table value))) globals
        have nextDecomposition : values = (before ++ [value]) ++ rest := by
          simpa only [List.append_assoc, List.singleton_append] using decomposition
        simpa only [List.length_append, List.length_singleton] using
          ih (before ++ [value]) nextDecomposition (bump table value) nextEnv
            (by simpa [nextEnv, setLocal] using getCounts)

/-- Counting leaves the complete caller heap intact and returns one fresh dictionary. -/
theorem result (heap : Mutable.Heap) (address : Nat) (values : List Int) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list (numbers values))) :
    ReturnsValue program "count_values" [.ref address] heap (.ref heap.length)
      (working heap (counts values)) globals := by
  let entry := functionEnv ((findFunctionIn program.body "count_values").getD ([], [])).1
    ((findFunctionIn program.body "count_values").getD ([], [])).2 [.ref address] globals
  let ready := setLocal entry "counts" (.ref heap.length)
  have getValues : ready "values" = some (.ref address) := by cases globals <;> rfl
  apply BlockReturns.function rfl rfl
  apply (BlockContinues.annAssign (Evaluates.dictionary .nil)).returns
  apply BlockReturns.for_sequence (items := numbers values)
    (by intro expressions impossible; cases impossible) (Evaluates.lookup getValues)
  · have preserved : (heap ++ [Mutable.Object.dict []])[address]? = some (.list (numbers values)) :=
      input_found heap address values found []
    simp [Mutable.sequence, preserved]
  · exact scan heap address values found values [] (by simp) [] ready globals rfl

theorem «Returns» (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    (HeapCalls.call program "count_values" [.ref address] heap).Returns := by
  obtain ⟨values, found⟩ := Mutable.Value.ints_of_hasType typed
  exact (result heap address values [] found).returns

theorem «Supported» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    (HeapCalls.call program "count_values" [.ref address] heap).Supported := («Returns» heap address valid typed).supported

end count_values

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

namespace player1_wins

/-- Tokens are separated by Python's Unicode whitespace rules. -/
def tokens (line : String) : List String := Strings.splitWhitespace (Strings.stripWhitespace line)

private abbrev text (value : String) : Mutable.Value := .scalar (.string value)
private abbrev values (line : String) : List Mutable.Value := (tokens line).map text
private abbrev splitHeap (heap : Mutable.Heap) (line : String) : Mutable.Heap := heap ++ [.list (values line)]

/-- Tokenization evaluates both string methods and allocates exactly one list. -/
theorem tokenize (line : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getLine : env "line" = some (text line)) :
    Evaluates program env (.call (.attribute (.call (.attribute (.name "line") "strip") []) "split") [])
      heap (.ref heap.length) (splitHeap heap line) globals := by
  apply Evaluates.split_whitespace
  · exact Evaluates.strip (Evaluates.lookup getLine) (by intro name equal; cases equal; simp [getLine])
  · intro name impossible
    cases impossible

private theorem empty_test (line : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getParts : env "parts" = some (.ref heap.length)) :
    Evaluates program env (.unaryOp .not (.name "parts")) (splitHeap heap line)
      (flag (tokens line).isEmpty) (splitHeap heap line) globals := by
  apply Evaluates.unary (Evaluates.lookup getParts)
  simp [HeapCalls.unary, Mutable.truth, splitHeap, values, flag]

/-- Empty and whitespace-only lines return false without calling either hand evaluator. -/
theorem blank (line : String) (heap : Mutable.Heap) (globals : Mutable.Globals) (empty : tokens line = []) :
    ReturnsValue program "player1_wins" [text line] heap (flag false) (heap ++ [.list []]) globals := by
  let entry := functionEnv ((findFunctionIn program.body "player1_wins").getD ([], [])).1
    ((findFunctionIn program.body "player1_wins").getD ([], [])).2 [text line] globals
  let ready := setLocal entry "parts" (.ref heap.length)
  have getLine : entry "line" = some (text line) := by cases globals <;> rfl
  have allocated : splitHeap heap line = heap ++ [.list []] := by simp [splitHeap, values, empty]
  rw [← allocated]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (tokenize line heap entry globals getLine)
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := splitHeap heap line)
  · simpa [empty] using empty_test line heap ready globals rfl
  · rfl
  · exact BlockReturns.return_value Evaluates.constant

/-- Every all-whitespace Unicode line takes the empty-line branch. -/
theorem whitespace (line : String) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (onlyWhitespace : line.toList.all Strings.isWhitespace = true) :
    ReturnsValue program "player1_wins" [text line] heap (flag false) (heap ++ [.list []]) globals := by
  have dropAll : ∀ characters : List Char, characters.all Strings.isWhitespace = true →
      characters.dropWhile Strings.isWhitespace = [] := by
    intro characters
    induction characters with
    | nil => intro _; rfl
    | cons character rest ih =>
        intro all
        have both := Bool.and_eq_true_iff.mp all
        rw [List.dropWhile_cons_of_pos both.1]
        exact ih both.2
  apply blank line heap globals
  have stripped : Strings.stripWhitespace line = "" := by
    simp [Strings.stripWhitespace, dropAll line.toList onlyWhitespace]
  rw [tokens, stripped]
  rfl

/-- The two slices are separate fresh lists, even for lines containing fewer than ten tokens. -/
def handsHeap (heap : Mutable.Heap) (line : String) : Mutable.Heap :=
  splitHeap heap line ++ [.list ((values line).take 5), .list ((values line).drop 5)]

private theorem first_hand (line : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getParts : env "parts" = some (.ref heap.length)) :
    Evaluates program env (.subscript (.name "parts") (.slice none (some (.constant (.int 5))) none))
      (splitHeap heap line) (.ref (heap.length+1)) (splitHeap heap line ++ [.list ((values line).take 5)]) globals := by
  apply Evaluates.slice (.cons (Evaluates.lookup getParts) (.cons Evaluates.constant
    (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
  have sliced := Mutable.getSlice_list_prefix (splitHeap heap line) heap.length (values line) 5 (by simp [splitHeap])
  have prefixValues : (values line).take (FormalSW.Py.Builtin.Slice.prefixLength (values line).length 5) =
      (values line).take 5 := by
    rw [show (5 : Int) = ((5 : Nat) : Int) from rfl, FormalSW.Py.Builtin.Slice.prefixLength_nonnegative]
    by_cases short : (values line).length ≤ 5
    · rw [Nat.min_eq_right short, List.take_length, List.take_of_length_le short]
    · rw [Nat.min_eq_left (by omega)]
  simpa only [prefixValues, splitHeap, List.length_append, List.length_singleton] using sliced

private theorem second_hand (line : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getParts : env "parts" = some (.ref heap.length)) :
    Evaluates program env (.subscript (.name "parts") (.slice (some (.constant (.int 5))) none none))
      (splitHeap heap line ++ [.list ((values line).take 5)]) (.ref (heap.length+2)) (handsHeap heap line) globals := by
  apply Evaluates.slice (.cons (Evaluates.lookup getParts) (.cons Evaluates.constant
    (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
  have retained : (splitHeap heap line ++ [.list ((values line).take 5)])[heap.length]? = some (.list (values line)) := by
    simp [splitHeap, List.append_assoc]
  have sliced := Mutable.getSlice_list_suffix (splitHeap heap line ++ [.list ((values line).take 5)]) heap.length (values line) 5 retained
  have size : (splitHeap heap line ++ [Mutable.Object.list ((values line).take 5)]).length = heap.length+2 := by
    simp only [splitHeap, List.length_append, List.length_singleton, Nat.add_assoc]
  have merged : splitHeap heap line ++ [.list ((values line).take 5)] ++ [.list ((values line).drop 5)] = handsHeap heap line := by
    rw [List.append_assoc]
    rfl
  rw [size, merged] at sliced
  exact sliced

/-- Once the rank calls and their comparison are proved, this composes the complete nonempty-line path. -/
theorem ranked (line : String) (heap : Mutable.Heap) (globals : Mutable.Globals) (nonempty : tokens line ≠ [])
    (left right : Mutable.Value) (middle finalHeap : Mutable.Heap) (wins : Bool)
    (first : ReturnsValue program "hand_rank" [.ref (heap.length+1)] (handsHeap heap line) left middle globals)
    (second : ReturnsValue program "hand_rank" [.ref (heap.length+2)] middle right finalHeap globals)
    (order : HeapCalls.compare .gt left right finalHeap = some (.ok (flag wins), finalHeap)) :
    ReturnsValue program "player1_wins" [text line] heap (flag wins) finalHeap globals := by
  let entry := functionEnv ((findFunctionIn program.body "player1_wins").getD ([], [])).1
    ((findFunctionIn program.body "player1_wins").getD ([], [])).2 [text line] globals
  let parsed := setLocal entry "parts" (.ref heap.length)
  let dealt := setLocal (setLocal parsed "h1" (.ref (heap.length+1))) "h2" (.ref (heap.length+2))
  have getLine : entry "line" = some (text line) := by cases globals <;> rfl
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (tokenize line heap entry globals getLine)
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := splitHeap heap line)
  · simpa [List.isEmpty_eq_false_iff.mpr nonempty] using empty_test line heap parsed globals rfl
  · rfl
  · apply BlockReturns.assign (first_hand line heap parsed globals rfl)
    apply BlockReturns.assign (second_hand line heap (setLocal parsed "h1" (.ref (heap.length+1))) globals (by simp [setLocal, parsed]))
    apply BlockReturns.return_value
    exact Evaluates.compare (env := dealt)
      (Evaluates.call (.cons (Evaluates.lookup (by simp [dealt, setLocal])) .nil) first)
      (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) second) order

/-- Different hand categories decide the winner without inspecting the tie-breaking fields. -/
theorem different_categories (line : String) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (nonempty : tokens line ≠ []) (a b : Nat) (firstCategory secondCategory : Int)
    (leftKeys rightKeys : List Mutable.Value) (middle finalHeap : Mutable.Heap)
    (first : ReturnsValue program "hand_rank" [.ref (heap.length+1)] (handsHeap heap line) (.ref a) middle globals)
    (second : ReturnsValue program "hand_rank" [.ref (heap.length+2)] middle (.ref b) finalHeap globals)
    (left : finalHeap[a]? = some (.tuple (number firstCategory :: leftKeys)))
    (right : finalHeap[b]? = some (.tuple (number secondCategory :: rightKeys)))
    (different : firstCategory ≠ secondCategory) :
    ReturnsValue program "player1_wins" [.scalar (.string line)] heap
      (flag (decide (secondCategory < firstCategory))) finalHeap globals := by
  exact ranked line heap globals nonempty (.ref a) (.ref b) middle finalHeap _ first second
    (compare_tuple_head_gt finalHeap a b firstCategory secondCategory leftKeys rightKeys left right different)

end player1_wins
end solvers.«54»
