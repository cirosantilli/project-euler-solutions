import «.formalsw».solvers.«62.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.KeyDictionaryCalls
import ProjectEulerSolutions.Lib.CubicPermutations
import ProjectEulerSolutions.Lib.HeapCollection

/-! Source proofs for the cubic-permutation solver. -/
set_option Elab.async false

namespace solvers.«62»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.CubicPermutations ProjectEulerSolutions.DigitHistogram
private abbrev program := Solvers.P62Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private abbrev character (digit : Nat) : Mutable.Value := .scalar (.string (String.singleton (Nat.digitChar digit)))
private def numbers (items : List Nat) : List Mutable.Value := items.map (fun (n : Nat) => number n)

namespace digit_signature
def indexExpression : Expr := .binOp (.builtinCall .ord [.name "ch"]) .sub (.constant (.int 48))
def body : List Stmt := [.augAssign (.subscript (.name "cnt") indexExpression) .add (.constant (.int 1))]
def code : List Stmt :=
  [.assign [.name "cnt"] (.binOp (.list [.constant (.int 0)]) .mult (.constant (.int 10))),
    .forIn (.name "ch") (.builtinCall .str [.name "x"]) body [], .return (.builtinCall .tuple [.name "cnt"])]
theorem code_source : executableBody ((findFunctionIn program.body "digit_signature").getD ([], [])).2 = code := rfl
private def table (digits : List Nat) : List Mutable.Value := numbers (counts digits)
private def working (base : Mutable.Heap) (digits : List Nat) : Mutable.Heap := base ++ [.list (table digits)]

theorem index_evaluate (digit : Nat) (bounded : digit < 10) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getCharacter : env "ch" = some (character digit)) :
    Evaluates program env indexExpression heap (number digit) heap := by
  have ordinal : Evaluates program env (.builtinCall .ord [.name "ch"]) heap
      (number (Nat.digitChar digit).toNat) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getCharacter) .nil)
    change scalarResult (some (Strings.ord (.string (String.singleton (Nat.digitChar digit))))) heap = _
    rw [String.singleton_eq_ofList, Strings.ord_character]
    rfl
  apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) ordinal Evaluates.constant
  simp [Mutable.binary, scalarBinary, number, digit_ordinal digit bounded]

theorem increment (digit : Nat) (bounded : digit < 10) (acc : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getCharacter : env "ch" = some (character digit)) (getCounts : env "cnt" = some (.ref base.length)) :
    BlockContinues program env body (working base acc) env (working base (acc ++ [digit])) := by
  have item : (table acc)[digit]? = some (number (acc.count digit)) := by simp [table, numbers, bounded]
  have changed : (working base acc).set base.length
      (.list ((table acc).set digit (number (acc.count digit + 1)))) = working base (acc ++ [digit]) := by
    simp only [working, table, numbers, append_digit, List.map_set]
    simp
  rw [← changed]
  apply BlockContinues.augItem (previous := number (acc.count digit)) (right := number 1)
    (updated := number (acc.count digit + 1)) (key := number digit)
    (afterReceiver := working base acc) (afterIndex := working base acc) (afterRead := working base acc)
    (afterRhs := working base acc) (afterOperation := working base acc)
    (by intro a b c impossible; cases impossible) (Evaluates.lookup getCounts)
  · exact index_evaluate digit bounded (working base acc) env getCharacter
  · have read := Mutable.getItem_list (working base acc) base.length digit (table acc)
      (by simp [working]) (by simpa [table, numbers] using bounded)
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using read
  · exact Evaluates.constant
  · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
  · simp [Mutable.setItem, working, Mutable.index, table, numbers, number, Except.map,
      show ¬ (digit : Int) < 0 by omega,
      Nat.not_le.mpr bounded]

theorem finish (base : Mutable.Heap) (acc : List Nat) (env : HeapCalls.Env)
    (getCounts : env "cnt" = some (.ref base.length)) :
    BlockReturns program env [.return (.builtinCall .tuple [.name "cnt"])] (working base acc)
      (.ref (base.length + 1)) (working base acc ++ [.tuple (table acc)]) := by
  apply BlockReturns.return_value
  apply Evaluates.builtin (.cons (Evaluates.lookup getCounts) .nil)
  have converted := primitive_tuple_list (working base acc) base.length (table acc) (by simp [working])
  simpa [Mutable.allocateTuple, working, table, numbers, counts, Except.map] using converted

private theorem sequence (n : Nat) (heap : Mutable.Heap) :
    Mutable.sequence (.scalar (.string (toString (n : Int)))) heap =
      some (.ok ((decimal n).map character), heap) := by
  change some ((Except.ok ((toString (n : Int)).toList.map
    (fun c => Mutable.Value.scalar (.string (String.singleton c)))) : Except ExceptionObject (List Mutable.Value)), heap) = _
  rw [text_decimal]
  simp only [List.map_map, Function.comp_def]

/-- Each source character updates the existing counter list before the string iterator advances. -/
theorem loop (n : Nat) (remaining : List Nat) (position : Nat) (suffix : (decimal n).drop position = remaining)
    (acc : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getCounts : env "cnt" = some (.ref base.length)) :
    BlockReturns program env
      [.sequenceStepLoop "ch" (.scalar (.string (toString (n : Int)))) position body [],
        .return (.builtinCall .tuple [.name "cnt"])] (working base acc) (.ref (base.length + 1))
      (working base (acc ++ remaining) ++ [.tuple (table (acc ++ remaining))]) := by
  induction remaining generalizing position acc env with
  | nil =>
      simp only [List.append_nil]
      apply BlockReturns.sequence_done (source := .scalar (.string (toString (n : Int))))
        (items := (decimal n).map character) (sequence n (working base acc))
      · have missing : (decimal n)[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := decimal n) (i := position)).symm
        simp [missing]
      · exact finish base acc env getCounts
  | cons digit rest ih =>
      have item : (decimal n)[position]? = some digit := by
        simpa [suffix] using (List.head?_drop (l := decimal n) (i := position)).symm
      have bounded := decimal_bounded n digit (List.mem_of_getElem? item)
      let bound := setLocal env "ch" (character digit)
      have boundCounts : bound "cnt" = some (.ref base.length) := by simpa [bound, setLocal] using getCounts
      have updated := increment digit bounded acc base bound rfl boundCounts
      have tailSuffix : (decimal n).drop (position + 1) = rest := by
        simpa [List.drop_drop, Nat.add_comm] using congrArg (List.drop 1) suffix
      have continued := ih (position + 1) tailSuffix (acc ++ [digit]) bound boundCounts
      apply BlockReturns.sequence_step (source := .scalar (.string (toString (n : Int))))
        (items := (decimal n).map character) (value := character digit) (sequence n (working base acc))
        (by simp [item])
      have full := updated.returns continued
      simpa only [Calls.loopContinuation, show Calls.loopControl body = false from rfl, Bool.false_eq_true,
        if_false, List.append_assoc, List.singleton_append] using full

/-- The helper's actual character scan returns its ten counters, including the string zero case. -/
theorem result (n : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "digit_signature" [number n] heap (.ref (heap.length + 2))
      (heap ++ [.list [number 0], .list (numbers (ProjectEulerSolutions.CubicPermutations.signature n)),
        .tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n))]) := by
  let base := heap ++ [Mutable.Object.list [number 0]]
  let entry := functionEnv ((findFunctionIn program.body "digit_signature").getD ([], [])).1
    ((findFunctionIn program.body "digit_signature").getD ([], [])).2 [number n] []
  let ready := setLocal entry "cnt" (.ref base.length)
  have finished := loop n (decimal n) 0 (by simp) [] base ready rfl
  have traversed : BlockReturns program ready (code.drop 1) (working base []) (.ref (base.length + 1))
      (working base (decimal n) ++ [.tuple (table (decimal n))]) := by
    apply BlockReturns.for_sequence (value := .scalar (.string (toString (n : Int))))
      (items := (decimal n).map character) (by intros; simp)
      (Evaluates.builtin (.cons (Evaluates.lookup (by simp [ready, entry, setLocal]; rfl)) .nil) rfl)
      (sequence n (working base []))
    simpa only [List.nil_append] using finished
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref base.length) (middle := working base [])
  · apply Evaluates.binary (av := .ref heap.length) (bv := number 10) (middle := base) (afterRight := base)
      (Evaluates.list (.cons Evaluates.constant .nil)) Evaluates.constant
    simp [Mutable.binary, Mutable.array, Mutable.allocate, base, working, table, numbers, number, counts,
      Except.map, show (BinOp.mult == BinOp.mult) = true from rfl]
  · simpa [ready, entry, base, working, table, ProjectEulerSolutions.CubicPermutations.signature, List.append_assoc, Nat.add_assoc] using traversed
end digit_signature

namespace groups
def condition : Expr := .membership (.name "sig") (.name "groups") false
def bucketExpression : Expr := .subscript (.name "groups") (.name "sig")
def minimumExpression : Expr := .subscript bucketExpression (.constant (.int 1))
def existing : List Stmt :=
  [.augAssign (.subscript bucketExpression (.constant (.int 0))) .add (.constant (.int 1)),
    .ifThen (.compare (.name "cube") [(.lt, minimumExpression)])
      [.assign [.subscript bucketExpression (.constant (.int 1))] (.name "cube")] []]
def fresh : List Stmt := [.assign [.subscript (.name "groups") (.name "sig")]
  (.list [.constant (.int 1), .name "cube"])]
private def sourceCode : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
def iteration : List Stmt := match (sourceCode[3]? : Option Stmt) with | some (.whileDo _ body _) => body | _ => []
theorem branch_source : iteration[4]? = some (.ifThen condition existing fresh) := rfl
def widthBody : List Stmt := match (iteration[2]? : Option Stmt) with | some (.ifThen _ body _) => body | _ => []
def clearExpression : Expr := .call (.attribute (.name "groups") "clear") []
def resetCode : List Stmt := [.expr clearExpression, .assign [.name "current_len"] (.name "length")]
theorem reset_source : widthBody.drop 2 = resetCode := rfl

theorem clear_keyed (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (entries : List (HashDict.Entry Mutable.Value)) (getGroups : env "groups" = some (.ref dictionary))
    (found : heap[dictionary]? = some (.keyDict entries)) :
    Evaluates program env clearExpression heap (.scalar .none) (heap.set dictionary (.keyDict [])) := by
  apply Evaluates.dictionary_clear (by intro name equal; cases equal; simp [getGroups])
    (Evaluates.lookup getGroups) (by simp [bindMethod, found])
  exact HeapCalls.clearDictionary_keyed heap dictionary entries found

theorem clear_scalar (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (entries : List (Constant × Mutable.Value)) (getGroups : env "groups" = some (.ref dictionary))
    (found : heap[dictionary]? = some (.dict entries)) :
    Evaluates program env clearExpression heap (.scalar .none) (heap.set dictionary (.dict [])) := by
  apply Evaluates.dictionary_clear (by intro name equal; cases equal; simp [getGroups])
    (Evaluates.lookup getGroups) (by simp [bindMethod, found])
  exact HeapCalls.clearDictionary_scalar heap dictionary entries found

/-- The original width transition clears the shared table and records the newly observed digit length. -/
theorem reset_keyed (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat) (length : Int)
    (entries : List (HashDict.Entry Mutable.Value)) (getGroups : env "groups" = some (.ref dictionary))
    (getLength : env "length" = some (number length)) (found : heap[dictionary]? = some (.keyDict entries)) :
    BlockContinues program env resetCode heap (setLocal env "current_len" (number length))
      (heap.set dictionary (.keyDict [])) := by
  exact (BlockContinues.discard (clear_keyed heap env dictionary entries getGroups found)).append
    (BlockContinues.assign (Evaluates.lookup getLength))

theorem hash_signature (n : Nat) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n)))) :
    Mutable.hashKey (.ref address) heap = some (.ok (key n), heap) := by
  have hashed := Mutable.hashKey_tuple_scalars heap address
    ((ProjectEulerSolutions.CubicPermutations.signature n).map (fun value : Nat => Constant.int value))
    (by simpa only [numbers, List.map_map, Function.comp_def] using found)
  simpa only [key, counterKey, List.map_map, Function.comp_def] using hashed

theorem membership_keyed (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary signatureAddress : Nat)
    (entries : List (HashDict.Entry Mutable.Value))
    (getGroups : env "groups" = some (.ref dictionary)) (getSignature : env "sig" = some (.ref signatureAddress))
    (found : heap[dictionary]? = some (.keyDict entries))
    (signatureFound : heap[signatureAddress]? = some (.tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n)))) :
    Evaluates program env condition heap (flag (HashDict.lookup entries (key n)).isSome) heap := by
  apply Evaluates.membership (.cons (Evaluates.lookup getSignature) (.cons (Evaluates.lookup getGroups) .nil))
  simp [HeapCalls.membership, found, hash_signature n heap signatureAddress signatureFound, flag, Except.map]

theorem membership_empty (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary signatureAddress : Nat)
    (getGroups : env "groups" = some (.ref dictionary)) (getSignature : env "sig" = some (.ref signatureAddress))
    (found : heap[dictionary]? = some (.dict []))
    (signatureFound : heap[signatureAddress]? = some (.tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n)))) :
    Evaluates program env condition heap (flag false) heap := by
  apply Evaluates.membership (.cons (Evaluates.lookup getSignature) (.cons (Evaluates.lookup getGroups) .nil))
  simp [HeapCalls.membership, found, hash_signature n heap signatureAddress signatureFound,
    Mutable.promoteDictionary, HashDict.lookup, HashDict.find?, flag, Except.map]

theorem bucket_evaluate (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary signatureAddress bucket : Nat)
    (entries : List (HashDict.Entry Mutable.Value))
    (getGroups : env "groups" = some (.ref dictionary)) (getSignature : env "sig" = some (.ref signatureAddress))
    (found : heap[dictionary]? = some (.keyDict entries))
    (signatureFound : heap[signatureAddress]? = some (.tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n))))
    (selected : HashDict.lookup entries (key n) = some (.ref bucket)) :
    Evaluates program env bucketExpression heap (.ref bucket) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
    (by intro a b c impossible; cases impossible) (Evaluates.lookup getGroups) (Evaluates.lookup getSignature)
  simp [Mutable.getItem, found, hash_signature n heap signatureAddress signatureFound, selected]

/-- A new group keeps the original tuple key and allocates a shared count/minimum bucket. -/
theorem insert_keyed (n : Nat) (cube : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary signatureAddress : Nat)
    (entries : List (HashDict.Entry Mutable.Value))
    (getGroups : env "groups" = some (.ref dictionary)) (getSignature : env "sig" = some (.ref signatureAddress))
    (getCube : env "cube" = some (number cube)) (found : heap[dictionary]? = some (.keyDict entries))
    (signatureFound : heap[signatureAddress]? = some (.tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n)))) :
    BlockContinues program env fresh heap env
      ((heap ++ [Mutable.Object.list [number 1, number cube]]).set dictionary
        (.keyDict (HashDict.insert entries (key n) (.ref signatureAddress) (.ref heap.length)))) := by
  let allocated := heap ++ [Mutable.Object.list [number 1, number cube]]
  have dictAfter : allocated[dictionary]? = some (.keyDict entries) :=
    ProjectEulerSolutions.HeapCollection.prefix_found heap _ dictionary _ found
  have signatureAfter : allocated[signatureAddress]? =
      some (.tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n))) :=
    ProjectEulerSolutions.HeapCollection.prefix_found heap _ signatureAddress _ signatureFound
  apply BlockContinues.setItem (value := .ref heap.length) (receiver := .ref dictionary)
    (subscript := .ref signatureAddress) (afterRhs := allocated) (afterTarget := allocated)
    (by intro a b c impossible; cases impossible)
    (Evaluates.list (.cons Evaluates.constant (.cons (Evaluates.lookup getCube) .nil)))
    (.cons (Evaluates.lookup getGroups) (.cons (Evaluates.lookup getSignature) .nil))
  simp [Mutable.setItem, dictAfter, hash_signature n allocated signatureAddress signatureAfter, Except.map, allocated]

/-- The first tuple insertion promotes the initially scalar-only empty dictionary. -/
theorem insert_empty (n : Nat) (cube : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary signatureAddress : Nat)
    (getGroups : env "groups" = some (.ref dictionary)) (getSignature : env "sig" = some (.ref signatureAddress))
    (getCube : env "cube" = some (number cube)) (found : heap[dictionary]? = some (.dict []))
    (signatureFound : heap[signatureAddress]? = some (.tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n)))) :
    BlockContinues program env fresh heap env
      ((heap ++ [Mutable.Object.list [number 1, number cube]]).set dictionary
        (.keyDict [(key n, .ref signatureAddress, .ref heap.length)])) := by
  let allocated := heap ++ [Mutable.Object.list [number 1, number cube]]
  have dictAfter : allocated[dictionary]? = some (.dict []) :=
    ProjectEulerSolutions.HeapCollection.prefix_found heap _ dictionary _ found
  have signatureAfter : allocated[signatureAddress]? =
      some (.tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n))) :=
    ProjectEulerSolutions.HeapCollection.prefix_found heap _ signatureAddress _ signatureFound
  apply BlockContinues.setItem (value := .ref heap.length) (receiver := .ref dictionary)
    (subscript := .ref signatureAddress) (afterRhs := allocated) (afterTarget := allocated)
    (by intro a b c impossible; cases impossible)
    (Evaluates.list (.cons Evaluates.constant (.cons (Evaluates.lookup getCube) .nil)))
    (.cons (Evaluates.lookup getGroups) (.cons (Evaluates.lookup getSignature) .nil))
  simp [Mutable.setItem, dictAfter, hash_signature n allocated signatureAddress signatureAfter,
    Mutable.promoteDictionary, HashDict.insert, Except.map, allocated]
end groups

theorem digit_signature_source_result (n : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "digit_signature" [number n] heap (.ref (heap.length + 2))
      (heap ++ [.list [number 0], .list (numbers (ProjectEulerSolutions.CubicPermutations.signature n)),
        .tuple (numbers (ProjectEulerSolutions.CubicPermutations.signature n))]) := digit_signature.result n heap

end solvers.«62»
