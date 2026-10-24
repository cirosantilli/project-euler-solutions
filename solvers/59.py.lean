import «.formalsw».solvers.«59.py.ast»
import ProjectEulerSolutions.Lib.HeapTables
import ProjectEulerSolutions.Lib.XorDecryption
import lang.py.FormalSW.DictionaryCalls
import lang.py.FormalSW.DictionaryItemsCalls
set_option Elab.async false
set_option maxRecDepth 4096


namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev program := Solvers.P59Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

/-- All fields describe pure scalar computations; no source-call behavior is assumed. -/
structure Letter where
  text : String
  value : Int
  lower : Nat
  upper : Nat
  ascii : Strings.isAscii text = true
  lower_ordinal : Strings.ord (.string text) = .ok (.int lower)
  upper_ordinal : Strings.ord (.string (Strings.asciiUpper text)) = .ok (.int upper)
  lower_bound : lower < 256
  upper_bound : upper < 256

def letterValues (entry : Letter) (items : List Mutable.Value) : List Mutable.Value :=
  (items.set entry.lower (number entry.value)).set entry.upper (number entry.value)
def lowerExpression : Expr := .builtinCall .ord [.name "ch"]
def upperExpression : Expr := .builtinCall .ord [.call (.attribute (.name "ch") "upper") []]
def letterBody : List Stmt :=
  [.assign [.subscript (.name "w") lowerExpression] (.name "val"),
   .assign [.subscript (.name "w") upperExpression] (.name "val")]

theorem letter_body_source :
    (executableBody ((findFunctionIn program.body "build_scoring_tables").getD ([], [])).2)[7]? =
      some (.forDictionaryItems (.tuple [.name "ch", .name "val"]) (.name "freq") letterBody []) := rfl

private theorem ordinal_evaluates (env : HeapCalls.Env) (operand : Expr) (heap middle : Mutable.Heap)
    (text : String) (index : Nat) (globals : Mutable.Globals)
    (receiver : Evaluates program env operand heap (.scalar (.string text)) middle globals)
    (computed : Strings.ord (.string text) = .ok (.int index)) :
    Evaluates program env (.builtinCall .ord [operand]) heap (number index) middle globals := by
  apply Evaluates.builtin (.cons receiver .nil)
  change HeapCalls.scalarResult (some (Strings.ord (.string text))) middle = _
  rw [computed]
  rfl

/-- A source letter cell runs both ord calls and the real .upper(), updating only the private weights list. -/
theorem letter_cell (entry : Letter) (base : Mutable.Heap) (items : List Mutable.Value)
    (garbage : Mutable.Heap) (capacity : 256 ≤ items.length) (env : HeapCalls.Env)
    (getW : env "w" = some (.ref base.length))
    (getCh : env "ch" = some (.scalar (.string entry.text)))
    (getVal : env "val" = some (number entry.value)) (globals : Mutable.Globals := []) :
    BlockContinues program env letterBody (HeapCollection.working base items garbage) env
      (HeapCollection.working base (letterValues entry items) garbage) globals := by
  let first := items.set entry.lower (number entry.value)
  have lower : Evaluates program env lowerExpression (HeapCollection.working base items garbage)
      (number entry.lower) (HeapCollection.working base items garbage) globals :=
    ordinal_evaluates env (.name "ch") _ _ entry.text entry.lower globals (Evaluates.lookup getCh) entry.lower_ordinal
  have writesLower : BlockContinues program env
      [.assign [.subscript (.name "w") lowerExpression] (.name "val")]
      (HeapCollection.working base items garbage) env (HeapCollection.working base first garbage) globals := by
    apply BlockContinues.setItem (value := number entry.value) (receiver := .ref base.length)
      (subscript := number entry.lower) (by intros; simp [lowerExpression]) (Evaluates.lookup getVal)
      (.cons (Evaluates.lookup getW) (.cons lower .nil))
    exact HeapTables.store base items garbage entry.lower (number entry.value) (by have := entry.lower_bound; omega)
  have upperText : Evaluates program env (.call (.attribute (.name "ch") "upper") [])
      (HeapCollection.working base first garbage) (.scalar (.string (Strings.asciiUpper entry.text)))
      (HeapCollection.working base first garbage) globals := by
    apply Evaluates.upper_ascii (Evaluates.lookup getCh) entry.ascii
    intro name equal
    cases equal
    rw [getCh]
    rfl
  have upper : Evaluates program env upperExpression (HeapCollection.working base first garbage)
      (number entry.upper) (HeapCollection.working base first garbage) globals :=
    ordinal_evaluates env (.call (.attribute (.name "ch") "upper") []) _ _
      (Strings.asciiUpper entry.text) entry.upper globals upperText entry.upper_ordinal
  have writesUpper : BlockContinues program env
      [.assign [.subscript (.name "w") upperExpression] (.name "val")]
      (HeapCollection.working base first garbage) env
      (HeapCollection.working base (first.set entry.upper (number entry.value)) garbage) globals := by
    apply BlockContinues.setItem (value := number entry.value) (receiver := .ref base.length)
      (subscript := number entry.upper) (by intros; simp [upperExpression]) (Evaluates.lookup getVal)
      (.cons (Evaluates.lookup getW) (.cons upper .nil))
    exact HeapTables.store base first garbage entry.upper (number entry.value)
      (by have := entry.upper_bound; simpa [first] using (show entry.upper < items.length by omega))
  exact writesLower.append writesUpper

end solvers.«59».build_scoring_tables
-- Append after the table-letters fragment.

namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev lettersModule := Solvers.P59Ast.module

def letters : List Letter := [
  ⟨"e", 6, 101, 69, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"t", 5, 116, 84, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"a", 5, 97, 65, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"o", 5, 111, 79, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"i", 4, 105, 73, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"n", 4, 110, 78, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"s", 4, 115, 83, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"h", 4, 104, 72, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"r", 4, 114, 82, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"d", 3, 100, 68, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"l", 3, 108, 76, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"u", 2, 117, 85, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"c", 2, 99, 67, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"m", 2, 109, 77, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"w", 2, 119, 87, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"f", 2, 102, 70, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"g", 2, 103, 71, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"y", 2, 121, 89, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"p", 2, 112, 80, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"b", 1, 98, 66, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"v", 1, 118, 86, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"k", 1, 107, 75, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"j", 0, 106, 74, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"x", 0, 120, 88, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"q", 0, 113, 81, by decide, by decide, by decide, by decide, by decide⟩,
  ⟨"z", 0, 122, 90, by decide, by decide, by decide, by decide, by decide⟩]

def frequencyConstants : List (Constant × Constant) :=
  letters.map (fun entry => (.string entry.text, .int entry.value))
def frequencyEntries : List (Constant × Mutable.Value) :=
  letters.map (fun entry => (.string entry.text, .scalar (.int entry.value)))
def frequencyExpression : Expr :=
  .dict (frequencyConstants.map (fun entry => (.constant entry.1, .constant entry.2)))

theorem frequency_source :
    (executableBody ((findFunctionIn lettersModule.body "build_scoring_tables").getD ([], [])).2)[6]? =
      some (.assign [.name "freq"] frequencyExpression) := rfl

theorem frequency_insert_order :
    frequencyConstants.foldl (fun table entry => ScalarDict.insert table entry.1 (.scalar entry.2)) [] =
      frequencyEntries := by decide

theorem frequency_literal (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    Evaluates lettersModule env frequencyExpression heap (.ref heap.length)
      (heap ++ [.dict frequencyEntries]) globals := by
  simpa only [frequency_insert_order] using
    (Evaluates.dictionary_constants (module := lettersModule) (env := env) (heap := heap)
      (globals := globals) frequencyConstants)

end solvers.«59».build_scoring_tables
-- Append after table-letters and table-letter-data fragments.

namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev dictionaryModule := Solvers.P59Ast.module

def entryFields (entry : Letter) : Constant × Mutable.Value :=
  (.string entry.text, .scalar (.int entry.value))
def entryPair (entry : Letter) : Mutable.Object :=
  .tuple [.scalar (.string entry.text), .scalar (.int entry.value)]
def pairObjects (entries : List Letter) : Mutable.Heap := entries.map entryPair
def letterFold (entries : List Letter) (items : List Mutable.Value) : List Mutable.Value :=
  entries.foldl (fun table entry => letterValues entry table) items

def dictionaryBody : List Stmt :=
  HeapCalls.comprehensionTargetBody (.tuple [.name "ch", .name "val"]) letterBody

def dictionaryKeys (entries : List Letter) : List SequenceSource :=
  entries.map (fun entry => .scalar (.string entry.text))

@[simp] theorem letterValues_length (entry : Letter) (items : List Mutable.Value) :
    (letterValues entry items).length = items.length := by simp [letterValues]

/-- Each real dictionary next allocates its pair, unpacks it, executes upper/ord and retains that tuple afterward. -/
theorem dictionary_loop (before remaining : List Letter) (base : Mutable.Heap)
    (items : List Mutable.Value) (garbage : Mutable.Heap) (capacity : 256 ≤ items.length)
    (address : Nat) (different : base.length ≠ address)
    (found : (HeapCollection.working base items garbage)[address]? =
      some (.dict ((before ++ remaining).map entryFields)))
    (env : HeapCalls.Env) (getW : env "w" = some (.ref base.length))
    (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      BlockContinues dictionaryModule env
        [.dictionaryItemsStepLoop "$formalsw.unpack" address (dictionaryKeys (before ++ remaining)) before.length dictionaryBody []]
        (HeapCollection.working base items garbage) afterEnv
        (HeapCollection.working base (letterFold remaining items) (garbage ++ pairObjects remaining)) globals ∧
      ∀ name, name ≠ "ch" → name ≠ "val" → name ≠ "$formalsw.unpack" → afterEnv name = env name := by
  induction remaining generalizing before items garbage env with
  | nil =>
      refine ⟨env, ?_, fun _ _ _ _ => rfl⟩
      simp only [letterFold, List.foldl_nil, pairObjects, List.map_nil, List.append_nil]
      apply BlockContinues.dictionary_items_done
      · simpa [dictionaryKeys, entryFields, List.map_map, Function.comp_def] using
          dictionaryItemsNext_dict (position := before.length) found
      · exact BlockContinues.nil
  | cons entry remaining ih =>
      let current := HeapCollection.working base items garbage
      let pairHeap := HeapCollection.working base items (garbage ++ [entryPair entry])
      let bound := HeapCalls.setLocal env "$formalsw.unpack" (.ref current.length)
      let unpacked := HeapCalls.assignNames bound ["ch", "val"]
        [.scalar (.string entry.text), .scalar (.int entry.value)]
      have fetched : dictionaryItemsNext address (dictionaryKeys (before ++ entry :: remaining)) before.length current =
          some (.ok (some (.scalar (.string entry.text), .scalar (.int entry.value))), current) := by
        simpa [dictionaryKeys, entryFields, List.map_map, Function.comp_def, current] using
          dictionaryItemsNext_dict (position := before.length) found
      have allocated : Mutable.allocateTuple [.scalar (.string entry.text), .scalar (.int entry.value)] current =
          some (.ok (.ref current.length), pairHeap) := by
        simp [Mutable.allocateTuple, pairHeap, current, entryPair, HeapCollection.append, Except.map]
      have pairShape : current ++ [entryPair entry] = pairHeap := HeapCollection.append base items garbage [entryPair entry]
      have pairFound : pairHeap[current.length]? = some (entryPair entry) := by
        rw [← pairShape]
        simp
      have unpack : BlockContinues dictionaryModule bound
          [.assign [.tuple [.name "ch", .name "val"]] (.name "$formalsw.unpack")]
          pairHeap unpacked pairHeap globals := by
        apply BlockContinues.unpack (names := ["ch", "val"])
          (items := [.scalar (.string entry.text), .scalar (.int entry.value)]) rfl
          (Evaluates.lookup rfl)
        · simp [Mutable.sequence, pairFound, entryPair]
        · rfl
      have unpackedW : unpacked "w" = some (.ref base.length) := getW
      have cell := letter_cell entry base items (garbage ++ [entryPair entry]) capacity unpacked unpackedW rfl rfl globals
      have rowRun : BlockContinues dictionaryModule bound dictionaryBody pairHeap unpacked
          (HeapCollection.working base (letterValues entry items) (garbage ++ [entryPair entry])) globals :=
        unpack.append cell
      have nextFound : (HeapCollection.working base (letterValues entry items) (garbage ++ [entryPair entry]))[address]? =
          some (.dict (((before ++ [entry]) ++ remaining).map entryFields)) := by
        have stable := HeapCollection.other_found base items (letterValues entry items) garbage [entryPair entry]
          address (.dict ((before ++ entry :: remaining).map entryFields)) different found
        simpa only [List.append_assoc, List.singleton_append] using stable
      obtain ⟨afterEnv, rest, stable⟩ := ih (before ++ [entry]) (letterValues entry items)
        (garbage ++ [entryPair entry]) (by simpa only [letterValues_length] using capacity) nextFound unpacked unpackedW
      refine ⟨afterEnv, ?_, ?_⟩
      · apply BlockContinues.dictionary_items_step fetched allocated
        simpa [dictionaryBody, HeapCalls.comprehensionTargetBody, letterFold, pairObjects,
          Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          dictionaryKeys, List.append_assoc, List.singleton_append, List.map_append] using rowRun.append rest
      · intro name notCh notVal notHidden
        exact (stable name notCh notVal notHidden).trans
          (by simp [unpacked, bound, HeapCalls.assignNames, HeapCalls.setLocal, notCh, notVal, notHidden])

/-- The actual .items receiver is captured once and its dictionary remains live throughout all26 writes. -/
theorem dictionary_result (entries : List Letter) (base : Mutable.Heap) (items : List Mutable.Value)
    (garbage : Mutable.Heap) (capacity : 256 ≤ items.length) (address : Nat) (different : base.length ≠ address)
    (found : (HeapCollection.working base items garbage)[address]? = some (.dict (entries.map entryFields)))
    (env : HeapCalls.Env) (getW : env "w" = some (.ref base.length))
    (getFreq : env "freq" = some (.ref address)) (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      BlockContinues dictionaryModule env
        [.forDictionaryItems (.tuple [.name "ch", .name "val"]) (.name "freq") letterBody []]
        (HeapCollection.working base items garbage) afterEnv
        (HeapCollection.working base (letterFold entries items) (garbage ++ pairObjects entries)) globals ∧
      ∀ name, name ≠ "ch" → name ≠ "val" → name ≠ "$formalsw.unpack" → afterEnv name = env name := by
  obtain ⟨afterEnv, executed, stable⟩ := dictionary_loop [] entries base items garbage capacity address different found env getW globals
  refine ⟨afterEnv, ?_, stable⟩
  apply BlockContinues.for_dictionary_items (Evaluates.lookup getFreq) found rfl
  simpa [dictionaryBody, dictionaryKeys, entryFields, List.map_map, Function.comp_def, HeapCalls.sequenceSource] using executed

end solvers.«59».build_scoring_tables

namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev prefixProgram := Solvers.P59Ast.module

def falseValue : Mutable.Value := .scalar (.bool false)
def trueValue : Mutable.Value := .scalar (.bool true)
def zeroValue : Mutable.Value := .scalar (.int 0)
def printableRaw : List Mutable.Value := List.replicate 256 falseValue
def printableRange : List Mutable.Value := HeapTables.fillRange 95 32 printableRaw trueValue
def printableBuilt : List Mutable.Value := (printableRange.set 10 trueValue).set 13 trueValue
def zeroWeights : List Mutable.Value := List.replicate 256 zeroValue
def spaceWeights : List Mutable.Value := zeroWeights.set 32 (.scalar (.int 10))

private theorem flatten_singletons (count : Nat) (value : Mutable.Value) :
    (List.replicate count [value]).flatten = List.replicate count value := by
  induction count with
  | zero => rfl
  | succ count ih => simp [List.replicate_succ, ih]

/-- The source literal and its repeated list are distinct retained allocations. -/
theorem allocate_repeated (program : Module) (value : Constant) (count : Nat)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    Evaluates program env (.binOp (.list [.constant value]) .mult (.constant (.int count))) heap
      (.ref (heap.length + 1)) (heap ++ [.list [.scalar value], .list (List.replicate count (.scalar value))]) globals := by
  let scratch := heap ++ [.list [.scalar value]]
  have literal : Evaluates program env (.list [.constant value]) heap (.ref heap.length) scratch globals :=
    Evaluates.list (.cons Evaluates.constant .nil)
  have operation : Mutable.binary .mult (.ref heap.length) (.scalar (.int count)) scratch =
      some (.ok (.ref (heap.length + 1)),
        heap ++ [.list [.scalar value], .list (List.replicate count (.scalar value))]) := by
    have mult : (BinOp.mult == BinOp.mult) = true := rfl
    simp only [Mutable.binary, mult, if_true]
    simp [Mutable.array, Mutable.allocate, scratch, List.append_assoc, Except.map]
  exact Evaluates.binary literal (Evaluates.constant (value := .int count)) operation

def printableCode : List Stmt :=
  [.assign [.name "printable"] (.binOp (.list [.constant (.bool false)]) .mult (.constant (.int 256))),
   .forIn (.name "v") (.call (.name "range") [.constant (.int 32), .constant (.int 127)])
      (HeapTables.fillBody "printable" "v" (.bool true)) [],
   .assign [.subscript (.name "printable") (.constant (.int 10))] (.constant (.bool true)),
   .assign [.subscript (.name "printable") (.constant (.int 13))] (.constant (.bool true))]

theorem printable_code_source :
    (executableBody ((findFunctionIn prefixProgram.body "build_scoring_tables").getD ([], [])).2).take 4 = printableCode := rfl

def printableHeap (caller : Mutable.Heap) : Mutable.Heap :=
  caller ++ [.list [falseValue], .list printableBuilt]

/-- The actual first four statements build printable using its95 range writes and two separate newline writes. -/
theorem printable_prefix (env : HeapCalls.Env) (caller : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      BlockContinues prefixProgram env printableCode caller afterEnv (printableHeap caller) globals ∧
      afterEnv "printable" = some (.ref (caller.length + 1)) := by
  let base := caller ++ [.list [falseValue]]
  let allocated := HeapCalls.setLocal env "printable" (.ref (caller.length + 1))
  have bootstrap : BlockContinues prefixProgram env
      [.assign [.name "printable"] (.binOp (.list [.constant (.bool false)]) .mult (.constant (.int 256)))]
      caller allocated (HeapCollection.working base printableRaw []) globals := by
    simpa [HeapCollection.working, base, falseValue, printableRaw, List.append_assoc] using
      (BlockContinues.assign (allocate_repeated prefixProgram (.bool false) 256 env caller globals))
  have getTable : allocated "printable" = some (.ref base.length) := by
    simp [allocated, base, HeapCalls.setLocal]
  obtain ⟨ranged, fills, stable⟩ := HeapTables.range_fill prefixProgram "printable" "v" (by decide)
    (.bool true) 95 32 127 (by decide) base printableRaw [] (by simp [printableRaw]) allocated getTable
    [.constant (.int 32), .constant (.int 127)] (.cons Evaluates.constant (.cons Evaluates.constant .nil))
  have rangedTable : ranged "printable" = some (.ref base.length) :=
    (stable "printable" (by decide)).trans getTable
  have tenWrite : BlockContinues prefixProgram ranged
      [.assign [.subscript (.name "printable") (.constant (.int 10))] (.constant (.bool true))]
      (HeapCollection.working base printableRange []) ranged
      (HeapCollection.working base (printableRange.set 10 trueValue) []) globals := by
    apply BlockContinues.setItem (value := trueValue) (receiver := .ref base.length)
      (subscript := .scalar (.int 10)) (by intros; simp) Evaluates.constant
      (.cons (Evaluates.lookup rangedTable) (.cons Evaluates.constant .nil))
    exact HeapTables.store base printableRange [] 10 trueValue (by simp [printableRange, printableRaw])
  have thirteenWrite : BlockContinues prefixProgram ranged
      [.assign [.subscript (.name "printable") (.constant (.int 13))] (.constant (.bool true))]
      (HeapCollection.working base (printableRange.set 10 trueValue) []) ranged
      (HeapCollection.working base printableBuilt []) globals := by
    apply BlockContinues.setItem (value := trueValue) (receiver := .ref base.length)
      (subscript := .scalar (.int 13)) (by intros; simp) Evaluates.constant
      (.cons (Evaluates.lookup rangedTable) (.cons Evaluates.constant .nil))
    exact HeapTables.store base (printableRange.set 10 trueValue) [] 13 trueValue
      (by simp [printableRange, printableRaw])
  refine ⟨ranged, ?_, ?_⟩
  · simpa [printableCode, printableHeap, HeapCollection.working, base, List.append_assoc] using
      bootstrap.append (fills.append (tenWrite.append thirteenWrite))
  · simpa [base] using rangedTable

end solvers.«59».build_scoring_tables
-- Append after table-prefix and the dictionary/data fragments.

namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev weightsProgram := Solvers.P59Ast.module

def weightsBase (caller : Mutable.Heap) : Mutable.Heap :=
  caller ++ [.list [falseValue], .list printableBuilt, .list [zeroValue]]
def letterHeap (caller : Mutable.Heap) : Mutable.Heap :=
  HeapCollection.working (weightsBase caller) (letterFold letters spaceWeights)
    ([.dict frequencyEntries] ++ pairObjects letters)

def weightsCode : List Stmt :=
  [.assign [.name "w"] (.binOp (.list [.constant (.int 0)]) .mult (.constant (.int 256))),
   .assign [.subscript (.name "w") (.builtinCall .ord [.constant (.string " ")])] (.constant (.int 10)),
   .assign [.name "freq"] frequencyExpression,
   .forDictionaryItems (.tuple [.name "ch", .name "val"]) (.name "freq") letterBody []]

theorem weights_code_source :
    ((executableBody ((findFunctionIn weightsProgram.body "build_scoring_tables").getD ([], [])).2).drop 4).take 4 =
      weightsCode := rfl

/-- Actual statements4..7 retain both weight allocations, the real dictionary and every iterator tuple. -/
theorem weights_prefix (env : HeapCalls.Env) (caller : Mutable.Heap)
    (getPrintable : env "printable" = some (.ref (caller.length + 1)))
    (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      BlockContinues weightsProgram env weightsCode (printableHeap caller) afterEnv (letterHeap caller) globals ∧
      afterEnv "printable" = some (.ref (caller.length + 1)) ∧
      afterEnv "w" = some (.ref (caller.length + 3)) := by
  let base := weightsBase caller
  let allocated := HeapCalls.setLocal env "w" (.ref (caller.length + 3))
  let spaced := HeapCollection.working base spaceWeights []
  let withFreq := HeapCalls.setLocal allocated "freq" (.ref (base.length + 1))
  have bootstrap : BlockContinues weightsProgram env
      [.assign [.name "w"] (.binOp (.list [.constant (.int 0)]) .mult (.constant (.int 256)))]
      (printableHeap caller) allocated (HeapCollection.working base zeroWeights []) globals := by
    simpa [printableHeap, weightsBase, HeapCollection.working, base, zeroValue, zeroWeights, List.append_assoc] using
      (BlockContinues.assign (allocate_repeated weightsProgram (.int 0) 256 env (printableHeap caller) globals))
  have allocatedW : allocated "w" = some (.ref base.length) := by
    simp [allocated, weightsBase, base, HeapCalls.setLocal]
  have spaceOrdinal : Evaluates weightsProgram allocated (.builtinCall .ord [.constant (.string " ")])
      (HeapCollection.working base zeroWeights []) (.scalar (.int 32))
      (HeapCollection.working base zeroWeights []) globals :=
    Evaluates.builtin (.cons Evaluates.constant .nil) rfl
  have storesSpace : BlockContinues weightsProgram allocated
      [.assign [.subscript (.name "w") (.builtinCall .ord [.constant (.string " ")])] (.constant (.int 10))]
      (HeapCollection.working base zeroWeights []) allocated spaced globals := by
    apply BlockContinues.setItem (value := .scalar (.int 10)) (receiver := .ref base.length)
      (subscript := .scalar (.int 32)) (by intros; simp) Evaluates.constant
      (.cons (Evaluates.lookup allocatedW) (.cons spaceOrdinal .nil))
    exact HeapTables.store base zeroWeights [] 32 (.scalar (.int 10)) (by simp [zeroWeights])
  have allocatesFrequency : BlockContinues weightsProgram allocated [.assign [.name "freq"] frequencyExpression]
      spaced withFreq (HeapCollection.working base spaceWeights [.dict frequencyEntries]) globals := by
    simpa [spaced, HeapCollection.working, List.append_assoc] using
      (BlockContinues.assign (frequency_literal allocated spaced globals))
  have getW : withFreq "w" = some (.ref base.length) := by
    simpa [withFreq, HeapCalls.setLocal] using allocatedW
  have getFreq : withFreq "freq" = some (.ref (base.length + 1)) := by simp [withFreq, HeapCalls.setLocal]
  have frequency_shape : frequencyEntries = letters.map entryFields := rfl
  have frequencyFound : (HeapCollection.working base spaceWeights [.dict frequencyEntries])[base.length + 1]? =
      some (.dict (letters.map entryFields)) := by
    rw [← frequency_shape]
    simp [HeapCollection.working]
  obtain ⟨afterEnv, writesLetters, stable⟩ := dictionary_result letters base spaceWeights [.dict frequencyEntries]
    (by simp [spaceWeights, zeroWeights]) (base.length + 1) (by omega) frequencyFound withFreq getW getFreq globals
  refine ⟨afterEnv, ?_, ?_, ?_⟩
  · simpa only [weightsCode, letterHeap, base, List.append_assoc, List.nil_append] using
      bootstrap.append (storesSpace.append (allocatesFrequency.append writesLetters))
  · exact (stable "printable" (by decide) (by decide) (by decide)).trans getPrintable
  · simpa [base, weightsBase] using (stable "w" (by decide) (by decide) (by decide)).trans getW

end solvers.«59».build_scoring_tables
-- Append after the table letter-data and dictionary-loop fragments.

namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst
open ProjectEulerSolutions

def printableValues : List Mutable.Value :=
  (List.range 256).map (fun (index : Nat) => .scalar (.bool (XorDecryption.printable index)))
def weightValues : List Mutable.Value :=
  (List.range 256).map (fun (index : Nat) => .scalar (.int (XorDecryption.charWeight index)))

/-- Exact retained scratch layout of the unmodified source helper. -/
def outputHeap (caller : Mutable.Heap) : Mutable.Heap :=
  caller ++ [.list [.scalar (.bool false)], .list printableValues,
    .list [.scalar (.int 0)], .list weightValues, .dict frequencyEntries] ++
    pairObjects letters ++ [.tuple [.ref (caller.length + 1), .ref (caller.length + 3)]]

/-- Caller-facing table facts needed by the streaming scorer; indices are genuine bytes. -/
structure Found (heap : Mutable.Heap) (printableAddress weightAddress : Nat) : Prop where
  printable : heap[printableAddress]? = some (.list printableValues)
  weights : heap[weightAddress]? = some (.list weightValues)

/-- The helper's return is an actual reference tuple, immediately unpackable by its caller. -/
def Result (caller : Mutable.Heap) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  value = .ref (caller.length + 31) ∧ after = outputHeap caller

@[simp] theorem printableValues_length : printableValues.length = 256 := by simp [printableValues]
@[simp] theorem weightValues_length : weightValues.length = 256 := by simp [weightValues]

end solvers.«59».build_scoring_tables

-- Append after the table result-model and dictionary-loop fragments.
namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst
open ProjectEulerSolutions

private abbrev tableInt (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev tableBool (value : Bool) : Mutable.Value := .scalar (.bool value)

def overwriteIndices (indices : List Nat) (replacement : Mutable.Value) (items : List Mutable.Value) :
    List Mutable.Value := indices.foldl (fun table index => table.set index replacement) items

def intAt (items : List Mutable.Value) (index : Nat) : Int :=
  match items[index]? with
  | some (.scalar (.int value)) => value
  | _ => 0

def decrementIndices (indices : List Nat) (amount : Int) (items : List Mutable.Value) : List Mutable.Value :=
  indices.foldl (fun table index => table.set index (tableInt (intAt table index - amount))) items

def punctuationIndices : List Nat := [46, 44, 59, 58, 39, 34, 33, 63, 40, 41, 45]
def penaltyIndices : List Nat := [123, 125, 91, 93, 124, 94, 126, 96]

def printableStage : List Mutable.Value :=
  ((HeapTables.fillRange 95 32 (List.replicate 256 (tableBool false)) (tableBool true)).set 10
    (tableBool true)).set 13 (tableBool true)
def letterStage : List Mutable.Value :=
  letterFold letters ((List.replicate 256 (tableInt 0)).set 32 (tableInt 10))
def punctuationStage : List Mutable.Value := overwriteIndices punctuationIndices (tableInt 1) letterStage
def digitsStage : List Mutable.Value := HeapTables.fillRange 10 48 punctuationStage (tableInt 0)
def penaltyStage : List Mutable.Value := decrementIndices penaltyIndices 3 digitsStage

abbrev actualPrintable := printableStage
abbrev actualWeights := penaltyStage

@[simp] theorem overwriteIndices_length (indices : List Nat) (value : Mutable.Value) (items : List Mutable.Value) :
    (overwriteIndices indices value items).length = items.length := by
  induction indices generalizing items with
  | nil => rfl
  | cons index indices ih => simpa only [overwriteIndices, List.foldl_cons, List.length_set] using ih (items.set index value)

@[simp] theorem decrementIndices_length (indices : List Nat) (amount : Int) (items : List Mutable.Value) :
    (decrementIndices indices amount items).length = items.length := by
  induction indices generalizing items with
  | nil => rfl
  | cons index indices ih =>
      simpa only [decrementIndices, List.foldl_cons, List.length_set] using
        ih (items.set index (tableInt (intAt items index - amount)))

@[simp] theorem letterFold_length (entries : List Letter) (items : List Mutable.Value) :
    (letterFold entries items).length = items.length := by
  induction entries generalizing items with
  | nil => rfl
  | cons entry entries ih => simpa only [letterFold, List.foldl_cons, letterValues_length] using ih (letterValues entry items)

/-- A functional view of a private 256-cell source list, independent of its heap address. -/
def TableModels (items : List Mutable.Value) (cells : Nat → Mutable.Value) : Prop :=
  items.length = 256 ∧ ∀ index, index < 256 → items[index]? = some (cells index)

def writeCell (cells : Nat → Mutable.Value) (position : Nat) (value : Mutable.Value) : Nat → Mutable.Value :=
  fun index => if index = position then value else cells index

theorem tableModels_set {items : List Mutable.Value} {cells : Nat → Mutable.Value}
    (model : TableModels items cells) (position : Nat) (value : Mutable.Value) (inside : position < 256) :
    TableModels (items.set position value) (writeCell cells position value) := by
  refine ⟨by simpa using model.1, ?_⟩
  intro index bound
  by_cases same : index = position
  · subst index
    rw [writeCell, if_pos rfl]
    exact List.getElem?_set_self (by rw [model.1]; exact inside)
  · simpa only [writeCell, if_neg same, List.getElem?_set_ne (Ne.symm same)] using model.2 index bound

theorem tableModels_fillRange {items : List Mutable.Value} {cells : Nat → Mutable.Value}
    (model : TableModels items cells) (count first : Nat) (value : Mutable.Value) (inside : first + count ≤ 256) :
    TableModels (HeapTables.fillRange count first items value)
      (fun index => if first ≤ index ∧ index < first + count then value else cells index) := by
  refine ⟨by simpa using model.1, ?_⟩
  intro index bound
  rw [HeapTables.fillRange_getElem? count first index items value (by rw [model.1]; exact inside)]
  by_cases changed : first ≤ index ∧ index < first + count
  · simp only [if_pos changed]
  · simp only [if_neg changed]
    exact model.2 index bound

theorem tableModels_replicate (value : Mutable.Value) :
    TableModels (List.replicate 256 value) (fun _ => value) := by
  refine ⟨by simp, ?_⟩
  intro index bound
  exact List.getElem?_replicate_of_lt bound

def letterCell (entry : Letter) (cells : Nat → Mutable.Value) : Nat → Mutable.Value :=
  writeCell (writeCell cells entry.lower (tableInt entry.value)) entry.upper (tableInt entry.value)
def letterCells (entries : List Letter) (cells : Nat → Mutable.Value) : Nat → Mutable.Value :=
  entries.foldl (fun cells entry => letterCell entry cells) cells

theorem tableModels_letterFold {items : List Mutable.Value} {cells : Nat → Mutable.Value}
    (model : TableModels items cells) (entries : List Letter) :
    TableModels (letterFold entries items) (letterCells entries cells) := by
  induction entries generalizing items cells with
  | nil => exact model
  | cons entry entries ih =>
      apply ih
      exact tableModels_set (tableModels_set model entry.lower (tableInt entry.value) entry.lower_bound)
        entry.upper (tableInt entry.value) entry.upper_bound

def overwriteCells (indices : List Nat) (replacement : Mutable.Value) (cells : Nat → Mutable.Value) : Nat → Mutable.Value :=
  indices.foldl (fun cells index => writeCell cells index replacement) cells

theorem tableModels_overwrite {items : List Mutable.Value} {cells : Nat → Mutable.Value}
    (model : TableModels items cells) (indices : List Nat) (value : Mutable.Value)
    (inside : ∀ index ∈ indices, index < 256) :
    TableModels (overwriteIndices indices value items) (overwriteCells indices value cells) := by
  induction indices generalizing items cells with
  | nil => exact model
  | cons index indices ih =>
      apply ih (tableModels_set model index value (inside index (by simp)))
      intro next present
      exact inside next (List.mem_cons_of_mem index present)

def scalarInt (value : Mutable.Value) : Int := match value with
  | .scalar (.int number) => number
  | _ => 0

def decrementCells (indices : List Nat) (amount : Int) (cells : Nat → Mutable.Value) : Nat → Mutable.Value :=
  indices.foldl (fun cells index => writeCell cells index (tableInt (scalarInt (cells index) - amount))) cells

theorem tableModels_decrement {items : List Mutable.Value} {cells : Nat → Mutable.Value}
    (model : TableModels items cells) (indices : List Nat) (amount : Int)
    (inside : ∀ index ∈ indices, index < 256) :
    TableModels (decrementIndices indices amount items) (decrementCells indices amount cells) := by
  induction indices generalizing items cells with
  | nil => exact model
  | cons index indices ih =>
      have read : intAt items index = scalarInt (cells index) := by
        simp only [intAt, model.2 index (inside index (by simp))]
        cases cells index with
        | scalar constant => cases constant <;> rfl
        | ref address => rfl
      apply ih
      · simpa only [read] using tableModels_set model index
          (tableInt (intAt items index - amount)) (inside index (by simp))
      · intro next present
        exact inside next (List.mem_cons_of_mem index present)

def printableCells : Nat → Mutable.Value :=
  writeCell (writeCell (fun index => if 32 ≤ index ∧ index < 127 then tableBool true else tableBool false)
    10 (tableBool true)) 13 (tableBool true)
def letterStageCells : Nat → Mutable.Value :=
  letterCells letters (writeCell (fun _ => tableInt 0) 32 (tableInt 10))
def punctuationCells : Nat → Mutable.Value := overwriteCells punctuationIndices (tableInt 1) letterStageCells
def digitsCells : Nat → Mutable.Value :=
  fun index => if 48 ≤ index ∧ index < 58 then tableInt 0 else punctuationCells index
def finalWeightCells : Nat → Mutable.Value := decrementCells penaltyIndices 3 digitsCells

theorem printableStage_models : TableModels printableStage printableCells := by
  exact tableModels_set
    (tableModels_set (tableModels_fillRange (tableModels_replicate (tableBool false)) 95 32
      (tableBool true) (by decide)) 10 (tableBool true) (by decide)) 13 (tableBool true) (by decide)

theorem letterStage_models : TableModels letterStage letterStageCells :=
  tableModels_letterFold (tableModels_set (tableModels_replicate (tableInt 0)) 32 (tableInt 10) (by decide)) letters

theorem punctuationStage_models : TableModels punctuationStage punctuationCells := by
  apply tableModels_overwrite letterStage_models
  decide

theorem digitsStage_models : TableModels digitsStage digitsCells :=
  tableModels_fillRange punctuationStage_models 10 48 (tableInt 0) (by decide)

theorem penaltyStage_models : TableModels penaltyStage finalWeightCells := by
  apply tableModels_decrement digitsStage_models
  decide

-- Sixteen small pure certificates follow. They evaluate only scalar lookup
-- functions (52 letter writes and the later updates), never any source trace
-- or a 256-cell mutable list. Each certificate checks 16 indices.
private theorem cells_chunk_00 : ∀ offset : Fin 16,
    printableCells (0 * 16 + offset.val) = tableBool (XorDecryption.printable (0 * 16 + offset.val)) ∧
    finalWeightCells (0 * 16 + offset.val) = tableInt (XorDecryption.charWeight (0 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_01 : ∀ offset : Fin 16,
    printableCells (1 * 16 + offset.val) = tableBool (XorDecryption.printable (1 * 16 + offset.val)) ∧
    finalWeightCells (1 * 16 + offset.val) = tableInt (XorDecryption.charWeight (1 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_02 : ∀ offset : Fin 16,
    printableCells (2 * 16 + offset.val) = tableBool (XorDecryption.printable (2 * 16 + offset.val)) ∧
    finalWeightCells (2 * 16 + offset.val) = tableInt (XorDecryption.charWeight (2 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_03 : ∀ offset : Fin 16,
    printableCells (3 * 16 + offset.val) = tableBool (XorDecryption.printable (3 * 16 + offset.val)) ∧
    finalWeightCells (3 * 16 + offset.val) = tableInt (XorDecryption.charWeight (3 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_04 : ∀ offset : Fin 16,
    printableCells (4 * 16 + offset.val) = tableBool (XorDecryption.printable (4 * 16 + offset.val)) ∧
    finalWeightCells (4 * 16 + offset.val) = tableInt (XorDecryption.charWeight (4 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_05 : ∀ offset : Fin 16,
    printableCells (5 * 16 + offset.val) = tableBool (XorDecryption.printable (5 * 16 + offset.val)) ∧
    finalWeightCells (5 * 16 + offset.val) = tableInt (XorDecryption.charWeight (5 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_06 : ∀ offset : Fin 16,
    printableCells (6 * 16 + offset.val) = tableBool (XorDecryption.printable (6 * 16 + offset.val)) ∧
    finalWeightCells (6 * 16 + offset.val) = tableInt (XorDecryption.charWeight (6 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_07 : ∀ offset : Fin 16,
    printableCells (7 * 16 + offset.val) = tableBool (XorDecryption.printable (7 * 16 + offset.val)) ∧
    finalWeightCells (7 * 16 + offset.val) = tableInt (XorDecryption.charWeight (7 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_08 : ∀ offset : Fin 16,
    printableCells (8 * 16 + offset.val) = tableBool (XorDecryption.printable (8 * 16 + offset.val)) ∧
    finalWeightCells (8 * 16 + offset.val) = tableInt (XorDecryption.charWeight (8 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_09 : ∀ offset : Fin 16,
    printableCells (9 * 16 + offset.val) = tableBool (XorDecryption.printable (9 * 16 + offset.val)) ∧
    finalWeightCells (9 * 16 + offset.val) = tableInt (XorDecryption.charWeight (9 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_10 : ∀ offset : Fin 16,
    printableCells (10 * 16 + offset.val) = tableBool (XorDecryption.printable (10 * 16 + offset.val)) ∧
    finalWeightCells (10 * 16 + offset.val) = tableInt (XorDecryption.charWeight (10 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_11 : ∀ offset : Fin 16,
    printableCells (11 * 16 + offset.val) = tableBool (XorDecryption.printable (11 * 16 + offset.val)) ∧
    finalWeightCells (11 * 16 + offset.val) = tableInt (XorDecryption.charWeight (11 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_12 : ∀ offset : Fin 16,
    printableCells (12 * 16 + offset.val) = tableBool (XorDecryption.printable (12 * 16 + offset.val)) ∧
    finalWeightCells (12 * 16 + offset.val) = tableInt (XorDecryption.charWeight (12 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_13 : ∀ offset : Fin 16,
    printableCells (13 * 16 + offset.val) = tableBool (XorDecryption.printable (13 * 16 + offset.val)) ∧
    finalWeightCells (13 * 16 + offset.val) = tableInt (XorDecryption.charWeight (13 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_14 : ∀ offset : Fin 16,
    printableCells (14 * 16 + offset.val) = tableBool (XorDecryption.printable (14 * 16 + offset.val)) ∧
    finalWeightCells (14 * 16 + offset.val) = tableInt (XorDecryption.charWeight (14 * 16 + offset.val)) := by
  decide

private theorem cells_chunk_15 : ∀ offset : Fin 16,
    printableCells (15 * 16 + offset.val) = tableBool (XorDecryption.printable (15 * 16 + offset.val)) ∧
    finalWeightCells (15 * 16 + offset.val) = tableInt (XorDecryption.charWeight (15 * 16 + offset.val)) := by
  decide

theorem tableCells_agree (index : Nat) (inside : index < 256) :
    printableCells index = tableBool (XorDecryption.printable index) ∧
    finalWeightCells index = tableInt (XorDecryption.charWeight index) := by
  have remainder : index % 16 < 16 := Nat.mod_lt _ (by decide)
  have division := Nat.div_add_mod index 16
  have chunk : index / 16 < 16 := by omega
  have cases : index / 16 = 0 ∨ index / 16 = 1 ∨ index / 16 = 2 ∨ index / 16 = 3 ∨ index / 16 = 4 ∨ index / 16 = 5 ∨ index / 16 = 6 ∨ index / 16 = 7 ∨ index / 16 = 8 ∨ index / 16 = 9 ∨ index / 16 = 10 ∨ index / 16 = 11 ∨ index / 16 = 12 ∨ index / 16 = 13 ∨ index / 16 = 14 ∨ index / 16 = 15 := by omega
  rcases cases with case0 | case1 | case2 | case3 | case4 | case5 | case6 | case7 | case8 | case9 | case10 | case11 | case12 | case13 | case14 | case15
  · have same : index = 0 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_00 ⟨index % 16, remainder⟩
  · have same : index = 1 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_01 ⟨index % 16, remainder⟩
  · have same : index = 2 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_02 ⟨index % 16, remainder⟩
  · have same : index = 3 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_03 ⟨index % 16, remainder⟩
  · have same : index = 4 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_04 ⟨index % 16, remainder⟩
  · have same : index = 5 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_05 ⟨index % 16, remainder⟩
  · have same : index = 6 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_06 ⟨index % 16, remainder⟩
  · have same : index = 7 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_07 ⟨index % 16, remainder⟩
  · have same : index = 8 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_08 ⟨index % 16, remainder⟩
  · have same : index = 9 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_09 ⟨index % 16, remainder⟩
  · have same : index = 10 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_10 ⟨index % 16, remainder⟩
  · have same : index = 11 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_11 ⟨index % 16, remainder⟩
  · have same : index = 12 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_12 ⟨index % 16, remainder⟩
  · have same : index = 13 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_13 ⟨index % 16, remainder⟩
  · have same : index = 14 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_14 ⟨index % 16, remainder⟩
  · have same : index = 15 * 16 + index % 16 := by omega
    rw [same]
    exact cells_chunk_15 ⟨index % 16, remainder⟩

 theorem tableModels_range (cells : Nat → Mutable.Value) :
    TableModels ((List.range 256).map cells) cells := by
  refine ⟨by simp, ?_⟩
  intro index inside
  simp only [List.getElem?_map, List.getElem?_range inside, Option.map_some]

theorem tableModels_ext {left right : List Mutable.Value} {leftCells rightCells : Nat → Mutable.Value}
    (leftModel : TableModels left leftCells) (rightModel : TableModels right rightCells)
    (agree : ∀ index, index < 256 → leftCells index = rightCells index) : left = right := by
  apply List.ext_getElem?
  intro index
  by_cases inside : index < 256
  · rw [leftModel.2 index inside, rightModel.2 index inside, agree index inside]
  · have leftEnd : left.length ≤ index := by rw [leftModel.1]; omega
    have rightEnd : right.length ≤ index := by rw [rightModel.1]; omega
    simp only [List.getElem?_eq_none leftEnd, List.getElem?_eq_none rightEnd]

theorem printableStage_eq : printableStage = printableValues :=
  tableModels_ext printableStage_models (tableModels_range _)
    (fun index inside => (tableCells_agree index inside).1)

theorem penaltyStage_eq : penaltyStage = weightValues :=
  tableModels_ext penaltyStage_models (tableModels_range _)
    (fun index inside => (tableCells_agree index inside).2)

theorem actualPrintable_eq : actualPrintable = printableValues := printableStage_eq
 theorem actualWeights_eq : actualWeights = weightValues := penaltyStage_eq

end solvers.«59».build_scoring_tables
-- Append after HeapTables and the table-equality fragment.


namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev byteTableProgram := Solvers.P59Ast.module
variable {globals : Mutable.Globals}

/-- All table cells are machine-independent Python integers. -/
def IntegerTable (items : List Mutable.Value) : Prop :=
  ∀ index, index < items.length → ∃ value : Int, items[index]? = some (.scalar (.int value))

theorem IntegerTable.set {items : List Mutable.Value} (integers : IntegerTable items)
    (index : Nat) (replacement : Int) : IntegerTable (items.set index (.scalar (.int replacement))) := by
  intro query inside
  have originalInside : query < items.length := by simpa only [List.length_set] using inside
  by_cases same : query = index
  · subst query
    exact ⟨replacement, List.getElem?_set_self originalInside⟩
  · obtain ⟨value, found⟩ := integers query originalInside
    exact ⟨value, by simpa only [List.getElem?_set_ne (Ne.symm same)] using found⟩

theorem IntegerTable.read {items : List Mutable.Value} (integers : IntegerTable items)
    (index : Nat) (inside : index < items.length) :
    items[index]? = some (.scalar (.int (intAt items index))) := by
  obtain ⟨value, found⟩ := integers index inside
  simp only [intAt, found]

/-- Actual saved-target read, scalar subtraction, and writeback for w[ch] -= amount. -/
theorem decrement_index (program : Module) (tableName indexName : String)
    (amount : Int) (base : Mutable.Heap) (items : List Mutable.Value) (garbage : Mutable.Heap)
    (position : Nat) (inside : position < items.length) (integers : IntegerTable items)
    (env : HeapCalls.Env) (getTable : env tableName = some (.ref base.length))
    (getIndex : env indexName = some (.scalar (.int position))) :
    BlockContinues program env
      [.augAssign (.subscript (.name tableName) (.name indexName)) .sub (.constant (.int amount))]
      (HeapCollection.working base items garbage) env
      (HeapCollection.working base (items.set position (.scalar (.int (intAt items position - amount)))) garbage)
      globals := by
  have read := integers.read position inside
  have actual : items[position] = .scalar (.int (intAt items position)) :=
    Option.some.inj ((List.getElem?_eq_getElem inside).symm.trans read)
  apply BlockContinues.augItem (target := .ref base.length) (key := .scalar (.int position))
    (previous := .scalar (.int (intAt items position))) (right := .scalar (.int amount))
    (updated := .scalar (.int (intAt items position - amount)))
    (by intros; simp) (Evaluates.lookup getTable) (Evaluates.lookup getIndex)
  · rw [Mutable.getItem_list _ _ _ _ (HeapCollection.found base items garbage) inside, actual]
  · exact Evaluates.constant
  · rfl
  · exact HeapTables.store base items garbage position (.scalar (.int (intAt items position - amount))) inside

theorem bytes_overwrite (program : Module) (tableName indexName : String) (different : tableName ≠ indexName)
    (replacement : Constant) (bytes : List UInt8) (base : Mutable.Heap) (items : List Mutable.Value)
    (garbage : Mutable.Heap) (capacity : 256 ≤ items.length) (env : HeapCalls.Env)
    (getTable : env tableName = some (.ref base.length)) :
    ∃ afterEnv,
      BlockContinues program env [.forIn (.name indexName) (.constant (.bytes bytes))
        (HeapTables.fillBody tableName indexName replacement) []]
        (HeapCollection.working base items garbage) afterEnv
        (HeapCollection.working base (overwriteIndices (bytes.map UInt8.toNat) (.scalar replacement) items) garbage) globals ∧
      ∀ name, name ≠ indexName → afterEnv name = env name := by
  have loop := HeapTables.bytes_result (globals := globals) (base := base) (garbage := garbage) program
    tableName indexName different (HeapTables.fillBody tableName indexName replacement) (by rfl)
    (fun byte table => table.set byte.toNat (.scalar replacement))
    (fun table => 256 ≤ table.length)
    (by intro byte table safe; simpa only [List.length_set] using safe)
    (by
      intro byte table current safe getTable getIndex
      apply HeapTables.write_index program tableName indexName replacement base table garbage byte.toNat
        (by have := byte.toNat_lt; omega) current getTable getIndex)
    bytes items capacity env getTable
  simpa only [HeapTables.byteFold, overwriteIndices, List.foldl_map] using loop

theorem bytes_decrement (program : Module) (tableName indexName : String) (different : tableName ≠ indexName)
    (amount : Int) (bytes : List UInt8) (base : Mutable.Heap) (items : List Mutable.Value)
    (garbage : Mutable.Heap) (capacity : 256 ≤ items.length) (integers : IntegerTable items)
    (env : HeapCalls.Env) (getTable : env tableName = some (.ref base.length)) :
    ∃ afterEnv,
      BlockContinues program env [.forIn (.name indexName) (.constant (.bytes bytes))
        [.augAssign (.subscript (.name tableName) (.name indexName)) .sub (.constant (.int amount))] []]
        (HeapCollection.working base items garbage) afterEnv
        (HeapCollection.working base (decrementIndices (bytes.map UInt8.toNat) amount items) garbage) globals ∧
      ∀ name, name ≠ indexName → afterEnv name = env name := by
  have loop := HeapTables.bytes_result (globals := globals) (base := base) (garbage := garbage) program
    tableName indexName different
    [.augAssign (.subscript (.name tableName) (.name indexName)) .sub (.constant (.int amount))] (by rfl)
    (fun byte table => table.set byte.toNat (.scalar (.int (intAt table byte.toNat - amount))))
    (fun table => 256 ≤ table.length ∧ IntegerTable table)
    (by
      intro byte table safe
      exact ⟨by simpa only [List.length_set] using safe.1, safe.2.set _ _⟩)
    (by
      intro byte table current safe getTable getIndex
      exact decrement_index program tableName indexName amount base table garbage byte.toNat
        (by have := byte.toNat_lt; omega) safe.2 current getTable getIndex)
    bytes items ⟨capacity, integers⟩ env getTable
  simpa only [HeapTables.byteFold, decrementIndices, List.foldl_map] using loop

end solvers.«59».build_scoring_tables
-- Append after table byte loops. Only pure table facts and small source AST anchors.
namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst
open ProjectEulerSolutions

private abbrev convenienceProgram := Solvers.P59Ast.module

theorem IntegerTable.replicate (count : Nat) (value : Int) :
    IntegerTable (List.replicate count (.scalar (.int value))) := by
  intro index inside
  have bound : index < count := by simpa only [List.length_replicate] using inside
  exact ⟨value, List.getElem?_replicate_of_lt bound⟩

theorem IntegerTable.letterFold {items : List Mutable.Value} (integers : IntegerTable items) (entries : List Letter) :
    IntegerTable (letterFold entries items) := by
  induction entries generalizing items with
  | nil => exact integers
  | cons entry entries ih =>
      exact ih ((integers.set entry.lower entry.value).set entry.upper entry.value)

theorem IntegerTable.overwrite {items : List Mutable.Value} (integers : IntegerTable items)
    (indices : List Nat) (value : Int) :
    IntegerTable (overwriteIndices indices (.scalar (.int value)) items) := by
  induction indices generalizing items with
  | nil => exact integers
  | cons index indices ih => exact ih (integers.set index value)

theorem IntegerTable.fillRange {items : List Mutable.Value} (integers : IntegerTable items)
    (count first : Nat) (value : Int) :
    IntegerTable (HeapTables.fillRange count first items (.scalar (.int value))) := by
  induction count generalizing first items with
  | zero => exact integers
  | succ count ih =>
      exact ih (items := items.set first (.scalar (.int value))) (first := first + 1) (integers.set first value)

theorem digitsStage_integer : IntegerTable digitsStage :=
  IntegerTable.fillRange
    (IntegerTable.overwrite (IntegerTable.letterFold ((IntegerTable.replicate 256 0).set 32 10) letters)
      punctuationIndices 1) 10 48 0

theorem digitsStage_capacity : 256 ≤ digitsStage.length := by
  exact Nat.le_of_eq digitsStage_models.1.symm

def punctuationBytes : List UInt8 := [46, 44, 59, 58, 39, 34, 33, 63, 40, 41, 45]
def penaltyBytes : List UInt8 := [123, 125, 91, 93, 124, 94, 126, 96]

@[simp] theorem punctuationBytes_indices : punctuationBytes.map UInt8.toNat = punctuationIndices := by decide
@[simp] theorem penaltyBytes_indices : penaltyBytes.map UInt8.toNat = penaltyIndices := by decide

theorem punctuation_source :
    (executableBody ((findFunctionIn convenienceProgram.body "build_scoring_tables").getD ([], [])).2)[8]? =
      some (.forIn (.name "ch") (.constant (.bytes punctuationBytes))
        (HeapTables.fillBody "w" "ch" (.int 1)) []) := rfl

theorem penalty_source :
    (executableBody ((findFunctionIn convenienceProgram.body "build_scoring_tables").getD ([], [])).2)[10]? =
      some (.forIn (.name "ch") (.constant (.bytes penaltyBytes))
        [.augAssign (.subscript (.name "w") (.name "ch")) .sub (.constant (.int 3))] []) := rfl

end solvers.«59».build_scoring_tables
-- Append after all table fragments, including pure equalities and byte convenience.

namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev tableProgram := Solvers.P59Ast.module

def tableGarbage : Mutable.Heap := [.dict frequencyEntries] ++ pairObjects letters

def tailCode : List Stmt :=
  [.forIn (.name "ch") (.constant (.bytes punctuationBytes))
      (HeapTables.fillBody "w" "ch" (.int 1)) [],
   .forIn (.name "v")
      (.call (.name "range") [.builtinCall .ord [.constant (.string "0")],
        .binOp (.builtinCall .ord [.constant (.string "9")]) .add (.constant (.int 1))])
      (HeapTables.fillBody "w" "v" (.int 0)) [],
   .forIn (.name "ch") (.constant (.bytes penaltyBytes))
      [.augAssign (.subscript (.name "w") (.name "ch")) .sub (.constant (.int 3))] [],
   .return (.tuple [.name "printable", .name "w"])]

theorem complete_code_source :
    executableBody ((findFunctionIn tableProgram.body "build_scoring_tables").getD ([], [])).2 =
      printableCode ++ weightsCode ++ tailCode := rfl

def completedHeap (caller : Mutable.Heap) : Mutable.Heap :=
  HeapCollection.working (weightsBase caller) penaltyStage tableGarbage

theorem letters_length : letters.length = 26 := rfl

theorem completedHeap_length (caller : Mutable.Heap) :
    (completedHeap caller).length = caller.length + 31 := by
  simp [completedHeap, HeapCollection.working, weightsBase, tableGarbage,
    pairObjects, letters_length]

theorem printableBuilt_eq : printableBuilt = printableValues := by
  exact printableStage_eq

theorem completedHeap_return (caller : Mutable.Heap) :
    completedHeap caller ++ [.tuple [.ref (caller.length + 1), .ref (caller.length + 3)]] =
      outputHeap caller := by
  simp only [completedHeap, HeapCollection.working, weightsBase, tableGarbage,
    printableBuilt_eq, penaltyStage_eq, outputHeap, falseValue, zeroValue,
    List.append_assoc, List.cons_append, List.nil_append]

/-- Actual punctuation iteration, ord-based digit range, and augmented penalty writes.
The caller's original heap remains the prefix of the retained private objects. -/
theorem tail (env : HeapCalls.Env) (caller : Mutable.Heap)
    (getPrintable : env "printable" = some (.ref (caller.length + 1)))
    (getWeights : env "w" = some (.ref (caller.length + 3)))
    (globals : Mutable.Globals := []) :
    BlockReturns tableProgram env tailCode (letterHeap caller)
      (.ref (caller.length + 31)) (outputHeap caller) globals := by
  let base := weightsBase caller
  have getTable : env "w" = some (.ref base.length) := by
    simpa [base, weightsBase] using getWeights
  have letterCapacity : 256 ≤ letterStage.length := by
    simp [letterStage]
  obtain ⟨punctuated, punctuateRun, punctuateFrame⟩ := bytes_overwrite
    (globals := globals) tableProgram "w" "ch" (by decide) (.int 1)
    punctuationBytes base letterStage tableGarbage letterCapacity env getTable
  have punctuateCode : BlockContinues tableProgram env
      [.forIn (.name "ch") (.constant (.bytes punctuationBytes))
        (HeapTables.fillBody "w" "ch" (.int 1)) []]
      (letterHeap caller) punctuated
      (HeapCollection.working base punctuationStage tableGarbage) globals := by
    simpa only [letterHeap, tableGarbage, base, letterStage, spaceWeights, zeroWeights,
      punctuationBytes_indices, punctuationStage] using punctuateRun
  have punctuatedW : punctuated "w" = some (.ref base.length) :=
    (punctuateFrame "w" (by decide)).trans getTable
  have punctuatedP : punctuated "printable" = some (.ref (caller.length + 1)) :=
    (punctuateFrame "printable" (by decide)).trans getPrintable
  let punctuatedHeap := HeapCollection.working base punctuationStage tableGarbage
  have zeroOrdinal : Evaluates tableProgram punctuated
      (.builtinCall .ord [.constant (.string "0")]) punctuatedHeap
      (.scalar (.int 48)) punctuatedHeap globals :=
    Evaluates.builtin (.cons Evaluates.constant .nil) rfl
  have nineOrdinal : Evaluates tableProgram punctuated
      (.builtinCall .ord [.constant (.string "9")]) punctuatedHeap
      (.scalar (.int 57)) punctuatedHeap globals :=
    Evaluates.builtin (.cons Evaluates.constant .nil) rfl
  have endOrdinal : Evaluates tableProgram punctuated
      (.binOp (.builtinCall .ord [.constant (.string "9")]) .add (.constant (.int 1)))
      punctuatedHeap (.scalar (.int 58)) punctuatedHeap globals := by
    exact Evaluates.binary nineOrdinal Evaluates.constant rfl
  have digitBounds : Evaluates.Arguments tableProgram punctuated
      [.builtinCall .ord [.constant (.string "0")],
        .binOp (.builtinCall .ord [.constant (.string "9")]) .add (.constant (.int 1))]
      punctuatedHeap [.scalar (.int 48), .scalar (.int 58)] punctuatedHeap globals :=
    .cons zeroOrdinal (.cons endOrdinal .nil)
  obtain ⟨digitEnv, digitRun, digitFrame⟩ := HeapTables.range_fill
    (globals := globals) tableProgram "w" "v" (by decide) (.int 0) 10 48 58 (by decide)
    base punctuationStage tableGarbage
    (by simp [punctuationStage, letterStage])
    punctuated punctuatedW _ digitBounds
  have digitW : digitEnv "w" = some (.ref base.length) :=
    (digitFrame "w" (by decide)).trans punctuatedW
  have digitP : digitEnv "printable" = some (.ref (caller.length + 1)) :=
    (digitFrame "printable" (by decide)).trans punctuatedP
  obtain ⟨penaltyEnv, penaltyRun, penaltyFrame⟩ := bytes_decrement
    (globals := globals) tableProgram "w" "ch" (by decide) 3 penaltyBytes
    base digitsStage tableGarbage digitsStage_capacity digitsStage_integer digitEnv digitW
  have penaltyCode : BlockContinues tableProgram digitEnv
      [.forIn (.name "ch") (.constant (.bytes penaltyBytes))
        [.augAssign (.subscript (.name "w") (.name "ch")) .sub (.constant (.int 3))] []]
      (HeapCollection.working base digitsStage tableGarbage) penaltyEnv
      (completedHeap caller) globals := by
    simpa only [penaltyBytes_indices, penaltyStage, completedHeap, base] using penaltyRun
  have penaltyW : penaltyEnv "w" = some (.ref (caller.length + 3)) := by
    simpa [base, weightsBase] using (penaltyFrame "w" (by decide)).trans digitW
  have penaltyP : penaltyEnv "printable" = some (.ref (caller.length + 1)) :=
    (penaltyFrame "printable" (by decide)).trans digitP
  have returnTuple : BlockReturns tableProgram penaltyEnv
      [.return (.tuple [.name "printable", .name "w"])]
      (completedHeap caller) (.ref (caller.length + 31)) (outputHeap caller) globals := by
    have actual := Evaluates.tuple (module := tableProgram) (env := penaltyEnv)
      (heap := completedHeap caller) (globals := globals)
      (.cons (Evaluates.lookup penaltyP) (.cons (Evaluates.lookup penaltyW) .nil)) (by simp)
    rw [completedHeap_length, completedHeap_return] at actual
    exact BlockReturns.return_value actual
  simpa only [tailCode, digitsStage, List.append_assoc, List.cons_append, List.nil_append] using
    punctuateCode.returns (digitRun.returns (penaltyCode.returns returnTuple))

/-- The unmodified helper executes its twelve statements and returns the two real tables.
All source allocations, including the singleton scratch lists and 26 items tuples, remain live. -/
theorem result (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapCalls.ReturnsValue tableProgram "build_scoring_tables" [] heap
      (.ref (heap.length + 31)) (outputHeap heap) globals := by
  let params := ((findFunctionIn tableProgram.body "build_scoring_tables").getD ([], [])).1
  let body := ((findFunctionIn tableProgram.body "build_scoring_tables").getD ([], [])).2
  let env := HeapCalls.functionEnv params body [] globals
  apply BlockReturns.function rfl rfl
  change BlockReturns tableProgram env (executableBody body) heap
    (.ref (heap.length + 31)) (outputHeap heap) globals
  rw [show executableBody body = printableCode ++ weightsCode ++ tailCode from complete_code_source]
  obtain ⟨printableEnv, printableRun, printableFound⟩ := printable_prefix env heap globals
  obtain ⟨letterEnv, letterRun, preservedPrintable, letterWeights⟩ :=
    weights_prefix printableEnv heap printableFound globals
  simpa only [List.append_assoc] using
    (printableRun.append letterRun).returns
      (tail letterEnv heap preservedPrintable letterWeights globals)

theorem outputHeap_length (heap : Mutable.Heap) : (outputHeap heap).length = heap.length + 32 := by
  rw [← completedHeap_return]
  simp only [List.length_append, List.length_singleton, completedHeap_length]

end solvers.«59».build_scoring_tables
-- Optional caller-facing facts, append after table-composition.
namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst
open ProjectEulerSolutions

theorem outputHeap_found (heap : Mutable.Heap) :
    Found (outputHeap heap) (heap.length + 1) (heap.length + 3) := by
  let initial := heap ++ [.list [.scalar (.bool false)], .list printableValues,
    .list [.scalar (.int 0)], .list weightValues, .dict frequencyEntries]
  have printable : initial[heap.length + 1]? = some (.list printableValues) := by
    dsimp only [initial]
    rw [List.getElem?_append_right (by omega)]
    simp
  have weights : initial[heap.length + 3]? = some (.list weightValues) := by
    dsimp only [initial]
    rw [List.getElem?_append_right (by omega)]
    simp
  constructor
  · exact HeapCollection.prefix_found (initial ++ pairObjects letters)
      [.tuple [.ref (heap.length + 1), .ref (heap.length + 3)]] _ _
      (HeapCollection.prefix_found initial (pairObjects letters) _ _ printable)
  · exact HeapCollection.prefix_found (initial ++ pairObjects letters)
      [.tuple [.ref (heap.length + 1), .ref (heap.length + 3)]] _ _
      (HeapCollection.prefix_found initial (pairObjects letters) _ _ weights)

theorem outputHeap_tuple (heap : Mutable.Heap) :
    (outputHeap heap)[heap.length + 31]? =
      some (.tuple [.ref (heap.length + 1), .ref (heap.length + 3)]) := by
  rw [← completedHeap_return, ← completedHeap_length heap]
  simp

theorem outputHeap_extends (heap : Mutable.Heap) :
    ∃ extra, outputHeap heap = heap ++ extra := by
  refine ⟨[.list [.scalar (.bool false)], .list printableValues,
    .list [.scalar (.int 0)], .list weightValues, .dict frequencyEntries] ++
    pairObjects letters ++ [.tuple [.ref (heap.length + 1), .ref (heap.length + 3)]], ?_⟩
  simp only [outputHeap, List.append_assoc]

end solvers.«59».build_scoring_tables

namespace solvers.«59».build_scoring_tables
open FormalSW.PyAst

theorem «Returns» : (HeapCalls.call Solvers.P59Ast.module "build_scoring_tables" [] []).Returns :=
  (result []).returns

theorem «Supported» : (HeapCalls.call Solvers.P59Ast.module "build_scoring_tables" [] []).Supported :=
  «Returns».supported

end solvers.«59».build_scoring_tables



namespace solvers.«59».best_key_and_sum.inner
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
open XorDecryption
private abbrev program := Solvers.P59Ast.module

def number (value : Int) : Mutable.Value := .scalar (.int value)
def flag (value : Bool) : Mutable.Value := .scalar (.bool value)
def modulus : Expr := .binOp (.name "i") .mod (.constant (.int 3))
def residue (value : Int) : Expr := .compare modulus [(.eq, .constant (.int value))]
def xorAssignment (name : String) : Stmt :=
  .assign [.name "p"] (.binOp (.name "c") .bitXor (.name name))
def selectPlain : Stmt := .ifThen (residue 0) [xorAssignment "key0"]
  [.ifThen (residue 1) [xorAssignment "key1"] [xorAssignment "key2"]]
def rejectCondition : Expr := .unaryOp .not (.subscript (.name "printable") (.name "p"))
def addTotal : Stmt := .augAssign (.name "s") .add (.name "p")
def addScore : Stmt := .augAssign (.name "score") .add (.subscript (.name "weights") (.name "p"))
def body : List Stmt :=
  [.assign [.name "c"] (.subscript (.name "cipher") (.name "i")), selectPlain,
   .ifThen rejectCondition [.assign [.name "valid"] (.constant (.bool false)), .breakLoop] [],
   addTotal, addScore]

private def thirdKeyBody : List Stmt :=
  match ((executableBody ((findFunctionIn program.body "best_key_and_sum").getD ([], [])).2)[5]? : Option Stmt) with
  | some (Stmt.forIn _ _ first _) => match first[0]? with
    | some (Stmt.forIn _ _ second _) => match second[0]? with
      | some (Stmt.forIn _ _ third _) => third
      | _ => []
    | _ => []
  | _ => []

theorem source : thirdKeyBody[4]? =
    some (.forIn (.name "i") (.call (.name "range") [.name "n"]) body []) := rfl

def changed : List String := ["i", "c", "p", "valid", "s", "score"]
def Frame (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ changed → after name = before name

theorem Frame.refl (env : HeapCalls.Env) : Frame env env := fun _ _ => rfl
theorem Frame.trans {first second third : HeapCalls.Env} (left : Frame first second)
    (right : Frame second third) : Frame first third :=
  fun name outside => (right name outside).trans (left name outside)
theorem Frame.updated (env : HeapCalls.Env) (name : String) (value : Mutable.Value)
    (written : name ∈ changed) : Frame env (HeapCalls.setLocal env name value) := by
  intro other outside
  have different : other ≠ name := by intro same; subst other; exact outside written
  simp [HeapCalls.setLocal, different]

structure Inputs (env : HeapCalls.Env) (cipher : List UInt8) (key : Key) (printable weights : Nat) : Prop where
  cipher_value : env "cipher" = some (.scalar (.bytes cipher))
  key0 : env "key0" = some (number key.1)
  key1 : env "key1" = some (number key.2.1)
  key2 : env "key2" = some (number key.2.2)
  n : env "n" = some (number cipher.length)
  printable_value : env "printable" = some (.ref printable)
  weights_value : env "weights" = some (.ref weights)

theorem Inputs.frame {before after : HeapCalls.Env} {cipher : List UInt8} {key : Key} {p w : Nat}
    (inputs : Inputs before cipher key p w) (frame : Frame before after) : Inputs after cipher key p w := by
  constructor
  · exact (frame "cipher" (by decide)).trans inputs.cipher_value
  · exact (frame "key0" (by decide)).trans inputs.key0
  · exact (frame "key1" (by decide)).trans inputs.key1
  · exact (frame "key2" (by decide)).trans inputs.key2
  · exact (frame "n" (by decide)).trans inputs.n
  · exact (frame "printable" (by decide)).trans inputs.printable_value
  · exact (frame "weights" (by decide)).trans inputs.weights_value

def KeyBounds (key : Key) : Prop :=
  (0 ≤ key.1 ∧ key.1 < 256) ∧ (0 ≤ key.2.1 ∧ key.2.1 < 256) ∧ (0 ≤ key.2.2 ∧ key.2.2 < 256)

theorem keyByte_bounds {key : Key} (bounds : KeyBounds key) (position : Nat) :
    0 ≤ keyByte key position ∧ keyByte key position < 256 := by
  by_cases zero : position % 3 = 0
  · simpa only [keyByte, if_pos zero] using bounds.1
  · by_cases one : position % 3 = 1
    · simpa only [keyByte, if_neg zero, if_pos one] using bounds.2.1
    · simpa only [keyByte, if_neg zero, if_neg one] using bounds.2.2

structure Running (env : HeapCalls.Env) (score total : Int) : Prop where
  score_value : env "score" = some (number score)
  total_value : env "s" = some (number total)
  valid : env "valid" = some (flag true)

/-- Rejected candidates retain their actual partial accumulators; callers observe only valid=False. -/
def Finished (env : HeapCalls.Env) : Option (Int × Int) → Prop
  | none => env "valid" = some (flag false)
  | some pair => Running env pair.1 pair.2

theorem xor_eq (a b : Int) : integerBitXor a b = Int.xor a b := by
  cases a <;> cases b <;> rfl

/-- Scalar bytes indexing uses the actual bounds-checked runtime operation. -/
theorem get_byte (heap : Mutable.Heap) (cipher : List UInt8) (position : Nat)
    (byte : UInt8) (found : cipher[position]? = some byte) :
    Mutable.getItem (.scalar (.bytes cipher)) (number position) heap =
      some (.ok (number byte.toNat), heap) := by
  have inside : position < cipher.length := (List.getElem?_eq_some_iff.mp found).1
  have nonnegative : ¬ (position : Int) < 0 := by omega
  have inRange : ¬ (cipher.length : Int) ≤ position := by omega
  simp [Mutable.getItem, number, Strings.get, FormalSW.Py.Builtin.Sequence.get?,
    FormalSW.Py.Builtin.Sequence.normalizeIndex, scalarNumber, nonnegative, inRange, found]

theorem get_printable (heap : Mutable.Heap) (p w : Nat)
    (tables : build_scoring_tables.Found heap p w) (plain : Int)
    (bounds : 0 ≤ plain ∧ plain < 256) :
    Mutable.getItem (.ref p) (number plain) heap =
      some (.ok (flag (printable plain)), heap) := by
  have cast : (plain.toNat : Int) = plain := Int.toNat_of_nonneg bounds.1
  have inside : plain.toNat < build_scoring_tables.printableValues.length := by
    simp only [build_scoring_tables.printableValues_length]
    omega
  simpa [build_scoring_tables.printableValues, number, flag, cast] using
    Mutable.getItem_list heap p plain.toNat build_scoring_tables.printableValues tables.printable inside

theorem get_weight (heap : Mutable.Heap) (p w : Nat)
    (tables : build_scoring_tables.Found heap p w) (plain : Int)
    (bounds : 0 ≤ plain ∧ plain < 256) :
    Mutable.getItem (.ref w) (number plain) heap =
      some (.ok (number (charWeight plain)), heap) := by
  have cast : (plain.toNat : Int) = plain := Int.toNat_of_nonneg bounds.1
  have inside : plain.toNat < build_scoring_tables.weightValues.length := by
    simp only [build_scoring_tables.weightValues_length]
    omega
  simpa [build_scoring_tables.weightValues, number, cast] using
    Mutable.getItem_list heap w plain.toNat build_scoring_tables.weightValues tables.weights inside

end solvers.«59».best_key_and_sum.inner
-- Append after inner-state.
namespace solvers.«59».best_key_and_sum.inner
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions XorDecryption
private abbrev cellProgram := Solvers.P59Ast.module

theorem residue_evaluates (env : HeapCalls.Env) (position target : Nat)
    (getI : env "i" = some (number position)) (heap : Mutable.Heap)
    (globals : Mutable.Globals := []) :
    Evaluates cellProgram env (residue target) heap
      (flag (decide (position % 3 = target))) heap globals := by
  have remainder : Evaluates cellProgram env modulus heap (number (position % 3)) heap globals := by
    apply Evaluates.binary (Evaluates.lookup getI) Evaluates.constant
    simp [Mutable.binary, number, scalarBinary]
  apply Evaluates.compare remainder Evaluates.constant
  simp [HeapCalls.compare, number, flag, scalarCompare, evalCmpOp]
  omega

theorem xor_assignment (env : HeapCalls.Env) (name : String) (byte keyValue : Int)
    (getC : env "c" = some (number byte)) (getKey : env name = some (number keyValue))
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    BlockContinues cellProgram env [xorAssignment name] heap
      (HeapCalls.setLocal env "p" (number (Int.xor byte keyValue))) heap globals := by
  apply BlockContinues.assign
  apply Evaluates.binary (Evaluates.lookup getC) (Evaluates.lookup getKey)
  simp [Mutable.binary, number, scalarBitXor, xor_eq]

/-- Both modulus tests and the selected XOR are evaluated in the real source order. -/
theorem select_plain (env : HeapCalls.Env) (cipher : List UInt8) (key : Key) (p w position : Nat)
    (byte : UInt8) (inputs : Inputs env cipher key p w)
    (getI : env "i" = some (number position)) (getC : env "c" = some (number byte.toNat))
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    BlockContinues cellProgram env [selectPlain] heap
      (HeapCalls.setLocal env "p" (number (Int.xor byte.toNat (keyByte key position)))) heap globals := by
  have zeroTest := residue_evaluates env position 0 getI heap globals
  have oneTest := residue_evaluates env position 1 getI heap globals
  by_cases zero : position % 3 = 0
  · have selected := xor_assignment env "key0" byte.toNat key.1 getC inputs.key0 heap globals
    have branch := BlockContinues.branch (yes := [xorAssignment "key0"])
      (no := [.ifThen (residue 1) [xorAssignment "key1"] [xorAssignment "key2"]])
      (by simpa only [zero, decide_true] using zeroTest) (by rfl) selected
    simpa only [selectPlain, keyByte, if_pos zero] using branch
  · by_cases one : position % 3 = 1
    · have selected := xor_assignment env "key1" byte.toNat key.2.1 getC inputs.key1 heap globals
      have innerBranch := BlockContinues.branch (yes := [xorAssignment "key1"])
        (no := [xorAssignment "key2"])
        (by simpa only [one, decide_true] using oneTest) (by rfl) selected
      have outerBranch := BlockContinues.branch (yes := [xorAssignment "key0"])
        (no := [.ifThen (residue 1) [xorAssignment "key1"] [xorAssignment "key2"]])
        (by simpa only [zero, decide_false] using zeroTest) (by rfl) innerBranch
      simpa only [selectPlain, keyByte, if_neg zero, if_pos one] using outerBranch
    · have selected := xor_assignment env "key2" byte.toNat key.2.2 getC inputs.key2 heap globals
      have innerBranch := BlockContinues.branch (yes := [xorAssignment "key1"])
        (no := [xorAssignment "key2"])
        (by simpa only [one, decide_false] using oneTest) (by rfl) selected
      have outerBranch := BlockContinues.branch (yes := [xorAssignment "key0"])
        (no := [.ifThen (residue 1) [xorAssignment "key1"] [xorAssignment "key2"]])
        (by simpa only [zero, decide_false] using zeroTest) (by rfl) innerBranch
      simpa only [selectPlain, keyByte, if_neg zero, if_neg one] using outerBranch

def decodedEnv (env : HeapCalls.Env) (byte : UInt8) (key : Key) (position : Nat) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal env "c" (number byte.toNat)) "p"
    (number (Int.xor byte.toNat (keyByte key position)))

theorem decodedEnv_frame (env : HeapCalls.Env) (byte : UInt8) (key : Key) (position : Nat) :
    Frame env (decodedEnv env byte key position) :=
  (Frame.updated env "c" _ (by decide)).trans (Frame.updated _ "p" _ (by decide))

theorem decode (env : HeapCalls.Env) (cipher : List UInt8) (key : Key) (p w position : Nat)
    (byte : UInt8) (found : cipher[position]? = some byte) (inputs : Inputs env cipher key p w)
    (getI : env "i" = some (number position)) (heap : Mutable.Heap)
    (globals : Mutable.Globals := []) :
    BlockContinues cellProgram env
      [.assign [.name "c"] (.subscript (.name "cipher") (.name "i")), selectPlain]
      heap (decodedEnv env byte key position) heap globals := by
  have reads : Evaluates cellProgram env (.subscript (.name "cipher") (.name "i"))
      heap (number byte.toNat) heap globals :=
    Evaluates.subscript (by intros; simp) (Evaluates.lookup inputs.cipher_value)
      (Evaluates.lookup getI) (get_byte heap cipher position byte found)
  have assigns := BlockContinues.assign (name := "c") reads
  let cEnv := HeapCalls.setLocal env "c" (number byte.toNat)
  have cInputs : Inputs cEnv cipher key p w := inputs.frame (Frame.updated _ "c" _ (by decide))
  have cI : cEnv "i" = some (number position) := by simpa [cEnv, HeapCalls.setLocal] using getI
  have cC : cEnv "c" = some (number byte.toNat) := by simp [cEnv, HeapCalls.setLocal]
  exact assigns.append (select_plain cEnv cipher key p w position byte cInputs cI cC heap globals)

theorem reject_evaluates (env : HeapCalls.Env) (cipher : List UInt8) (key : Key) (p w : Nat)
    (plain : Int) (bounds : 0 ≤ plain ∧ plain < 256) (inputs : Inputs env cipher key p w)
    (getP : env "p" = some (number plain)) (heap : Mutable.Heap)
    (tables : build_scoring_tables.Found heap p w) (globals : Mutable.Globals := []) :
    Evaluates cellProgram env rejectCondition heap (flag (!(printable plain))) heap globals := by
  apply Evaluates.unary
    (Evaluates.subscript (by intros; simp) (Evaluates.lookup inputs.printable_value)
      (Evaluates.lookup getP) (get_printable heap p w tables plain bounds))
  simp [HeapCalls.unary, Mutable.truth, scalarTruth, flag]

/-- The accepted branch performs both source augmented assignments, preserving the old total until its update. -/
theorem accumulate (env : HeapCalls.Env) (cipher : List UInt8) (key : Key) (p w : Nat)
    (plain score total : Int) (bounds : 0 ≤ plain ∧ plain < 256) (inputs : Inputs env cipher key p w)
    (getP : env "p" = some (number plain)) (running : Running env score total)
    (heap : Mutable.Heap) (tables : build_scoring_tables.Found heap p w)
    (globals : Mutable.Globals := []) :
    BlockContinues cellProgram env [addTotal, addScore] heap
      (HeapCalls.setLocal (HeapCalls.setLocal env "s" (number (total + plain))) "score"
        (number (score + charWeight plain))) heap globals := by
  have totalRun : BlockContinues cellProgram env [addTotal] heap
      (HeapCalls.setLocal env "s" (number (total + plain))) heap globals := by
    exact BlockContinues.augAssign running.total_value (Evaluates.lookup getP) rfl
  let totalEnv := HeapCalls.setLocal env "s" (number (total + plain))
  have getScore : totalEnv "score" = some (number score) := by
    simpa [totalEnv, HeapCalls.setLocal] using running.score_value
  have getPlain : totalEnv "p" = some (number plain) := by
    simpa [totalEnv, HeapCalls.setLocal] using getP
  have getWeights : totalEnv "weights" = some (.ref w) := by
    simpa [totalEnv, HeapCalls.setLocal] using inputs.weights_value
  have readWeight : Evaluates cellProgram totalEnv (.subscript (.name "weights") (.name "p"))
      heap (number (charWeight plain)) heap globals :=
    Evaluates.subscript (by intros; simp) (Evaluates.lookup getWeights) (Evaluates.lookup getPlain)
      (get_weight heap p w tables plain bounds)
  exact totalRun.append (BlockContinues.augAssign getScore readWeight rfl)

end solvers.«59».best_key_and_sum.inner
-- Append after inner-state and inner-cell.
namespace solvers.«59».best_key_and_sum.inner
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions XorDecryption
private abbrev loopProgram := Solvers.P59Ast.module

/-- A branch may transfer through its pending loop boundary; its suffix stays explicit. -/
theorem branch_pending {module : Module} {env afterEnv : HeapCalls.Env}
    {condition : Expr} {yes no pending : List Stmt} {heap middle after : Mutable.Heap}
    {value : Mutable.Value} {tested : Bool} {globals : Mutable.Globals}
    (evaluated : Evaluates module env condition heap value middle globals)
    (truth : Mutable.truth value middle = some (.ok tested, middle))
    (rest : BlockContinues module env ((if tested then yes else no) ++ pending)
      middle afterEnv after globals) :
    BlockContinues module env (.ifThen condition yes no :: pending) heap afterEnv after globals := by
  intro following anchors stack
  exact .next rfl ((evaluated anchors _).trans
    (.next (following := ⟨.block (((if tested then yes else no) ++ pending) ++ following) env, stack, middle⟩)
      (by simp [step, run, truth, List.append_assoc]) (rest following anchors stack)))

/-- Actual break discards both pending accumulations and exits the nearest entered loop. -/
theorem break_pending (env : HeapCalls.Env) (next : Stmt) (heap : Mutable.Heap)
    (globals : Mutable.Globals := []) :
    BlockContinues loopProgram env [.breakLoop, addTotal, addScore, .loopResume next]
      heap env heap globals := by
  intro following anchors stack
  exact .next rfl .refl

theorem Running.bind_index {env : HeapCalls.Env} {score total : Int}
    (running : Running env score total) (position : Nat) :
    Running (HeapCalls.setLocal env "i" (number position)) score total := by
  constructor
  · simpa [HeapCalls.setLocal] using running.score_value
  · simpa [HeapCalls.setLocal] using running.total_value
  · simpa [HeapCalls.setLocal] using running.valid

theorem Running.decode {env : HeapCalls.Env} {score total : Int}
    (running : Running env score total) (byte : UInt8) (key : Key) (position : Nat) :
    Running (decodedEnv env byte key position) score total := by
  constructor
  · simpa [decodedEnv, HeapCalls.setLocal] using running.score_value
  · simpa [decodedEnv, HeapCalls.setLocal] using running.total_value
  · simpa [decodedEnv, HeapCalls.setLocal] using running.valid

def accumulatedEnv (env : HeapCalls.Env) (plain score total : Int) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal env "s" (number (total + plain))) "score"
    (number (score + charWeight plain))

theorem accumulatedEnv_frame (env : HeapCalls.Env) (plain score total : Int) :
    Frame env (accumulatedEnv env plain score total) :=
  (Frame.updated env "s" _ (by decide)).trans (Frame.updated _ "score" _ (by decide))

theorem Running.accumulate {env : HeapCalls.Env} {score total : Int}
    (running : Running env score total) (plain : Int) :
    Running (accumulatedEnv env plain score total) (score + charWeight plain) (total + plain) := by
  constructor
  · simp [accumulatedEnv, HeapCalls.setLocal]
  · simp [accumulatedEnv, HeapCalls.setLocal]
  · simpa [accumulatedEnv, HeapCalls.setLocal] using running.valid

/-- Structural recursion follows only the unconsumed ciphertext. A rejected byte executes break,
leaves its actual partial accumulators intact, and exposes valid=False to the candidate selector. -/
theorem range_loop (before remaining : List UInt8) (key : Key) (bounds : KeyBounds key)
    (p w : Nat) (heap : Mutable.Heap) (tables : build_scoring_tables.Found heap p w)
    (env : HeapCalls.Env) (inputs : Inputs env (before ++ remaining) key p w)
    (score total : Int) (running : Running env score total) (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      BlockContinues loopProgram env
        [.rangeStepLoop "i" before.length (before ++ remaining).length 1 body []]
        heap afterEnv heap globals ∧
      Finished afterEnv (scoreStreaming key before.length (encodeBytes remaining) score total) ∧
      Frame env afterEnv := by
  induction remaining generalizing before env score total with
  | nil =>
      refine ⟨env, ?_, running, Frame.refl env⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
  | cons byte remaining ih =>
      let cipher := before ++ byte :: remaining
      let position := before.length
      let boundEnv := HeapCalls.setLocal env "i" (number position)
      let plain := Int.xor byte.toNat (keyByte key position)
      let decoded := decodedEnv boundEnv byte key position
      let next := Stmt.rangeStepLoop "i" ((position : Int) + 1) cipher.length 1 body []
      have boundFrame : Frame env boundEnv := Frame.updated env "i" _ (by decide)
      have boundInputs : Inputs boundEnv cipher key p w := inputs.frame boundFrame
      have boundRunning : Running boundEnv score total := running.bind_index position
      have getI : boundEnv "i" = some (number position) := by simp [boundEnv, HeapCalls.setLocal]
      have byteFound : cipher[position]? = some byte := by
        dsimp only [cipher, position]
        rw [List.getElem?_append_right (Nat.le_refl _)]
        simp
      have decodedRun := decode boundEnv cipher key p w position byte byteFound boundInputs getI heap globals
      have decodedFrame : Frame boundEnv decoded := decodedEnv_frame boundEnv byte key position
      have decodedInputs : Inputs decoded cipher key p w := boundInputs.frame decodedFrame
      have decodedRunning : Running decoded score total := boundRunning.decode byte key position
      have plainBounds : 0 ≤ plain ∧ plain < 256 :=
        xor_byte_bounds (by have := byte.toNat_lt; constructor <;> omega) (keyByte_bounds bounds position)
      have getPlain : decoded "p" = some (number plain) := by
        simp [decoded, decodedEnv, plain, HeapCalls.setLocal]
      have guardRun := reject_evaluates decoded cipher key p w plain plainBounds decodedInputs getPlain heap tables globals
      by_cases accepted : printable plain = true
      · let accumulated := accumulatedEnv decoded plain score total
        have accumFrame : Frame decoded accumulated := accumulatedEnv_frame decoded plain score total
        have accumInputs : Inputs accumulated cipher key p w := decodedInputs.frame accumFrame
        have accumRunning : Running accumulated (score + charWeight plain) (total + plain) :=
          decodedRunning.accumulate plain
        have accumRun := accumulate decoded cipher key p w plain score total plainBounds decodedInputs
          getPlain decodedRunning heap tables globals
        have restInputs : Inputs accumulated ((before ++ [byte]) ++ remaining) key p w := by
          simpa only [List.append_assoc, List.singleton_append] using accumInputs
        obtain ⟨afterEnv, restRun, finished, restFrame⟩ := ih (before ++ [byte]) accumulated restInputs
          (score + charWeight plain) (total + plain) accumRunning
        have nextRun : BlockContinues loopProgram accumulated [next] heap afterEnv heap globals := by
          simpa only [next, position, cipher, List.length_append, List.length_singleton,
            List.append_assoc, List.singleton_append, Int.natCast_add, Int.natCast_one] using restRun
        have continuation : BlockContinues loopProgram decoded
            [addTotal, addScore, .loopResume next] heap afterEnv heap globals :=
          accumRun.append (BlockContinues.loop_resume nextRun)
        have guard : BlockContinues loopProgram decoded
            [.ifThen rejectCondition [.assign [.name "valid"] (.constant (.bool false)), .breakLoop] [],
              addTotal, addScore, .loopResume next] heap afterEnv heap globals := by
          apply branch_pending (tested := false) guardRun (by simp [flag, accepted, Mutable.truth, scalarTruth])
          exact continuation
        refine ⟨afterEnv, ?_, ?_, boundFrame.trans (decodedFrame.trans (accumFrame.trans restFrame))⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            next, position, cipher] using decodedRun.append guard
        · simpa only [scoreStreaming, encodeBytes, List.map_cons, plain, position,
            if_pos accepted, List.length_append, List.length_singleton] using finished
      · have rejected : printable plain = false := by cases actual : printable plain <;> simp_all
        let rejectedEnv := HeapCalls.setLocal decoded "valid" (flag false)
        have rejectFrame : Frame decoded rejectedEnv := Frame.updated decoded "valid" _ (by decide)
        have rejectRun : BlockContinues loopProgram decoded
            [.assign [.name "valid"] (.constant (.bool false)), .breakLoop,
              addTotal, addScore, .loopResume next] heap rejectedEnv heap globals :=
          (BlockContinues.assign Evaluates.constant).append (break_pending rejectedEnv next heap globals)
        have guard : BlockContinues loopProgram decoded
            [.ifThen rejectCondition [.assign [.name "valid"] (.constant (.bool false)), .breakLoop] [],
              addTotal, addScore, .loopResume next] heap rejectedEnv heap globals := by
          apply branch_pending (tested := true) guardRun (by simp [flag, rejected, Mutable.truth, scalarTruth])
          exact rejectRun
        refine ⟨rejectedEnv, ?_, ?_, boundFrame.trans (decodedFrame.trans rejectFrame)⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            next, position, cipher] using decodedRun.append guard
        · simp only [scoreStreaming, encodeBytes, List.map_cons, plain, position, if_neg accepted, Finished]
          simp [rejectedEnv, HeapCalls.setLocal]

/-- One real three-byte key candidate, with arbitrary heap/globals and no candidate-space expansion. -/
theorem result (cipher : List UInt8) (key : Key) (bounds : KeyBounds key)
    (p w : Nat) (heap : Mutable.Heap) (tables : build_scoring_tables.Found heap p w)
    (env : HeapCalls.Env) (inputs : Inputs env cipher key p w) (running : Running env 0 0)
    (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      BlockContinues loopProgram env
        [.forIn (.name "i") (.call (.name "range") [.name "n"]) body []]
        heap afterEnv heap globals ∧
      Finished afterEnv (scoreKeyStreaming (encodeBytes cipher) key) ∧ Frame env afterEnv := by
  obtain ⟨afterEnv, loop, finished, frame⟩ := range_loop [] cipher key bounds p w heap tables env inputs 0 0 running globals
  refine ⟨afterEnv, ?_, finished, frame⟩
  exact BlockContinues.for_range (.cons (Evaluates.lookup inputs.n) .nil) rfl loop

end solvers.«59».best_key_and_sum.inner
