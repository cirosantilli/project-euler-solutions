import solvers.«92.py.lean.d».Table

namespace solvers.«92».tableCalls
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := table.number

abbrev keyExpression : Expr := .tuple
  [.binOp (.name "total") .add (.binOp (.name "digit") .mult (.name "digit")),
   .boolOp .and [.name "tight", .compare (.name "digit") [(.eq, .name "ceiling")]]]

abbrev update : Stmt := .assign [.subscript (.name "following") (.name "key")]
  (.binOp (.call (.attribute (.name "following") "get") [.name "key", .constant (.int 0)])
    .add (.name "ways"))

abbrev body : List Stmt := [.assign [.name "key"] keyExpression, update]

theorem source :
    (match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[2]? with
      | some (Stmt.forIn _ _ prefixBody _) => match prefixBody[2]? with
          | some (Stmt.forDictionaryItems _ _ rowBody _) => match rowBody[0]? with
              | some (Stmt.forIn _ _ digitBody _) => digitBody
              | _ => []
          | _ => []
      | _ => []) = body := rfl

theorem key_result (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (state : Prefix.State) (digit ceiling : Nat)
    (getTotal : locals "total" = some (number state.1))
    (getTight : locals "tight" = some (.scalar (.bool state.2)))
    (getDigit : locals "digit" = some (number digit))
    (getCeiling : locals "ceiling" = some (number ceiling)) :
    Evaluates program locals keyExpression heap (.ref heap.length)
      (heap ++ [table.tuple (Prefix.advance ceiling state digit)]) globals := by
  apply Evaluates.tuple (values := [number (state.1 + digit * digit),
    .scalar (.bool (state.2 && decide (digit = ceiling)))]) _ (by simp)
  apply Evaluates.Arguments.cons (middle := heap)
  · apply Evaluates.binary (Evaluates.lookup getTotal)
      (Evaluates.binary (value := number (digit * digit)) (finalHeap := heap)
        (Evaluates.lookup getDigit) (Evaluates.lookup getDigit)
        (by simp [Mutable.binary, scalarBinary, number]))
    simp [Mutable.binary, scalarBinary, number]
  · apply Evaluates.Arguments.cons _ .nil
    have compared : Evaluates program locals (.compare (.name "digit") [(.eq, .name "ceiling")])
        heap (.scalar (.bool (decide (digit = ceiling)))) heap globals := by
      apply Evaluates.compare (Evaluates.lookup getDigit) (Evaluates.lookup getCeiling)
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    cases tight : state.2 with
    | false =>
        simpa only [tight, Bool.false_and] using
          (Evaluates.boolean_stop (op := .and) (flag := false)
            (rest := [.compare (.name "digit") [(.eq, .name "ceiling")]])
            (Evaluates.lookup getTight) (by simp [Mutable.truth, scalarTruth, tight])
            (by simp))
    | true =>
        simpa only [tight, Bool.true_and] using
          (Evaluates.boolean_next (op := .and) (flag := true) (Evaluates.lookup getTight)
            (by simp [Mutable.truth, scalarTruth, tight]) (by simp)
            (Evaluates.boolean_last compared rfl))

theorem get_result (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (address keyAddress : Nat) (entries : table.Entries) (state : Prefix.State)
    (found : table.Found heap address entries)
    (keyFound : heap[keyAddress]? = some (table.tuple state))
    (getDict : locals "following" = some (.ref address))
    (getKey : locals "key" = some (.ref keyAddress)) :
    Evaluates program locals
      (.call (.attribute (.name "following") "get") [.name "key", .constant (.int 0)])
      heap (number ((table.lookup entries state).getD 0)) heap globals := by
  apply Evaluates.dictionary_get_bound
    (by intro query same; cases same; simp [getDict]) (Evaluates.lookup getDict)
  · rcases found with found | ⟨_, found⟩ <;> simp [bindMethod, found]
  · exact .cons (Evaluates.lookup getKey) (.cons Evaluates.constant .nil)
  · exact table.get heap address keyAddress entries state found keyFound

theorem update_result (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (address keyAddress ways : Nat) (entries : table.Entries) (state : Prefix.State)
    (found : table.Found heap address entries)
    (keyFound : heap[keyAddress]? = some (table.tuple state))
    (getDict : locals "following" = some (.ref address))
    (getKey : locals "key" = some (.ref keyAddress))
    (getWays : locals "ways" = some (number ways)) :
    BlockContinues program locals [update] heap locals
      (heap.set address (.keyDict (table.encode (table.add entries state keyAddress ways)))) globals := by
  apply BlockContinues.setItem (by intro a b c; simp)
    (Evaluates.binary
      (value := number ((table.lookup entries state).getD 0 + ways)) (finalHeap := heap)
      (get_result heap locals globals address keyAddress entries state found keyFound getDict getKey)
      (Evaluates.lookup getWays)
      (by simp [Mutable.binary, scalarBinary, number]))
    (.cons (Evaluates.lookup getDict) (.cons (Evaluates.lookup getKey) .nil))
  exact table.set heap address keyAddress _ entries state found keyFound

/-- One source iteration allocates its key tuple and adds exactly one state's contribution. -/
theorem body_result (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (address ways digit ceiling : Nat) (entries : table.Entries) (state : Prefix.State)
    (found : table.Found heap address entries)
    (getDict : locals "following" = some (.ref address))
    (getWays : locals "ways" = some (number ways))
    (getTotal : locals "total" = some (number state.1))
    (getTight : locals "tight" = some (.scalar (.bool state.2)))
    (getDigit : locals "digit" = some (number digit))
    (getCeiling : locals "ceiling" = some (number ceiling)) :
    BlockContinues program locals body heap (setLocal locals "key" (.ref heap.length))
      ((heap ++ [table.tuple (Prefix.advance ceiling state digit)]).set address
        (.keyDict (table.encode (table.add entries (Prefix.advance ceiling state digit) heap.length ways))))
      globals := by
  apply BlockContinues.append
    (BlockContinues.assign (key_result heap locals globals state digit ceiling
      getTotal getTight getDigit getCeiling))
  apply update_result
  · exact found.append _
  · simp
  · simpa [setLocal] using getDict
  · simp [setLocal]
  · simpa [setLocal] using getWays

end solvers.«92».tableCalls
