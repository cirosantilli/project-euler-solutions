import solvers.«98.py.lean.d».SquareBounds
import ProjectEulerSolutions.Lib.PatternBuckets
import ProjectEulerSolutions.Lib.AnagramSearch

namespace solvers.«98».squares_with_length
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open PatternBuckets (Bucket View Partition)
open pattern_signature (program number)
open AnagramSearch (squareText)
set_option Elab.async false

def signature (text : String) : PatternBuckets.Key := AnagramSquares.labels [] text.toList
def setValues (squares : List String) : List Constant := (squares.map Constant.string).foldl ScalarSet.insert []

abbrev squareBody : List Stmt :=
  [.assign [.name "sq"] (.binOp (.name "n") .mult (.name "n")),
   .assign [.name "s"] (.builtinCall .str [.name "sq"]),
   .expr (.call (.attribute (.name "st") "add") [.name "s"]),
   .assign [.name "pat"] (.call (.name "pattern_signature") [.name "s"]),
   .expr PatternBuckets.insertion]

theorem source_loop : (((findFunctionIn program.body "squares_with_length").getD ([], [])).2)[6]? = some
    (.forIn (.name "n") (.call (.name "range") [.name "n0", .binOp (.name "n1") .add (.constant (.int 1))]) squareBody []) := rfl

theorem setValues_append (squares : List String) (square : String) :
    setValues (squares ++ [square]) = ScalarSet.insert (setValues squares) (.string square) := by
  simp [setValues, List.foldl_append]

/-- One actual source iteration inserts the printed square into both its pattern bucket and membership set. -/
theorem square_step (base suffix : Mutable.Heap) (dictionary setAddress : Nat) (buckets : List Bucket)
    (squares : List String) (root : Nat) (env : HeapCalls.Env)
    (view : View (base ++ suffix) dictionary buckets) (partition : Partition signature squares buckets)
    (foundSet : (base ++ suffix)[setAddress]? = some (.set (setValues squares)))
    (afterDictionary : base.length ≤ dictionary) (afterSet : base.length ≤ setAddress)
    (getDictionary : env "by_pat" = some (.ref dictionary)) (getSet : env "st" = some (.ref setAddress))
    (getRoot : env "n" = some (number root)) :
    ∃ finalEnv extra newBuckets,
      BlockContinues program env squareBody (base ++ suffix) finalEnv (base ++ extra) ∧
      View (base ++ extra) dictionary newBuckets ∧ Partition signature (squares ++ [squareText root]) newBuckets ∧
      (base ++ extra)[setAddress]? = some (.set (setValues (squares ++ [squareText root]))) ∧
      (∀ name, name ∉ ["sq", "s", "pat"] → finalEnv name = env name) := by
  let squared := setLocal env "sq" (number (root * root))
  let printed := setLocal squared "s" (PatternBuckets.text (squareText root))
  let newSet := setValues (squares ++ [squareText root])
  let setSuffix := suffix.set (setAddress - base.length) (.set newSet)
  have setHeap : (base ++ suffix).set setAddress (.set newSet) = base ++ setSuffix :=
    List.set_append_right setAddress (.set newSet) afterSet
  have multiplied : Evaluates program env (.binOp (.name "n") .mult (.name "n"))
      (base ++ suffix) (number (root * root)) (base ++ suffix) :=
    Evaluates.binary (av := number root) (bv := number root) (Evaluates.lookup getRoot) (Evaluates.lookup getRoot) rfl
  have rendered : Evaluates program squared (.builtinCall .str [.name "sq"])
      (base ++ suffix) (PatternBuckets.text (squareText root)) (base ++ suffix) := by
    apply Evaluates.builtin (.cons (Evaluates.lookup (show squared "sq" = some (number (root * root)) by simp [squared, setLocal])) .nil)
    simp [primitive, Strings.str, Strings.convert, squareText, PatternBuckets.text]
    rfl
  have getS : printed "s" = some (PatternBuckets.text (squareText root)) := by simp [printed, setLocal]
  have inserted := Evaluates.add (module := program) (globals := [])
    (Evaluates.lookup (show printed "st" = some (.ref setAddress) by simpa [printed, squared, setLocal] using getSet)) foundSet
    (Evaluates.lookup getS) foundSet
  have insertedSet : BlockContinues program printed [.expr (.call (.attribute (.name "st") "add") [.name "s"])]
      (base ++ suffix) printed (base ++ setSuffix) := by
    simpa only [PatternBuckets.text, ← setValues_append, ← setHeap, newSet] using BlockContinues.discard inserted
  have setView : View (base ++ setSuffix) dictionary buckets := by
    rw [← setHeap]
    exact view.write_set setAddress (setValues squares) newSet foundSet
  have setStored : (base ++ setSuffix)[setAddress]? = some (.set newSet) := by
    rw [← setHeap]
    exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp foundSet).1
  obtain ⟨patternHeap, patternAddress, patterned, patternFound, prefixHeap⟩ := pattern_signature.result (squareText root) (base ++ setSuffix)
  obtain ⟨patternExtra, rfl⟩ := prefixHeap
  let prepared := setLocal printed "pat" (.ref patternAddress)
  let newBuckets := PatternBuckets.push buckets (signature (squareText root)) patternAddress ((base ++ setSuffix) ++ patternExtra).length (squareText root)
  have patternView := setView.append patternExtra
  have bucketed := PatternBuckets.evaluate (globals := []) program prepared ((base ++ setSuffix) ++ patternExtra) dictionary buckets patternView
    (signature (squareText root)) patternAddress (squareText root)
    (by simpa [prepared, printed, squared, setLocal] using getDictionary)
    (by simp [prepared, setLocal]) (by simpa [prepared, setLocal] using getS) patternFound
  obtain ⟨extra, updatedHeap⟩ := PatternBuckets.updated_prefix base (setSuffix ++ patternExtra) dictionary buckets
    (by simpa only [List.append_assoc] using patternView) afterDictionary (signature (squareText root)) patternAddress (squareText root)
  have finalView := PatternBuckets.updated_view ((base ++ setSuffix) ++ patternExtra) dictionary buckets patternView
    (signature (squareText root)) patternAddress (squareText root)
  have finalSet := PatternBuckets.updated_set ((base ++ setSuffix) ++ patternExtra) dictionary buckets patternView
    (signature (squareText root)) patternAddress (squareText root) setAddress newSet
    (HeapCollection.prefix_found (base ++ setSuffix) patternExtra setAddress _ setStored)
  have finalPartition := partition.push patternAddress ((base ++ setSuffix) ++ patternExtra).length (squareText root)
  refine ⟨prepared, extra, newBuckets, ?_, ?_, finalPartition, ?_, ?_⟩
  · have executed := (BlockContinues.assign multiplied).append ((BlockContinues.assign rendered).append
      (insertedSet.append ((BlockContinues.assign (Evaluates.call (.cons (Evaluates.lookup getS) .nil) patterned)).append
        (BlockContinues.discard bucketed))))
    simpa only [squareBody, prepared, printed, squared, List.append_assoc, updatedHeap] using executed
  · simpa only [newBuckets, List.append_assoc, updatedHeap] using finalView
  · simpa only [List.append_assoc, updatedHeap] using finalSet
  · intro name different
    simp only [List.mem_cons, not_or] at different
    simp [prepared, printed, squared, setLocal, different.1, different.2.1, different.2.2]

end solvers.«98».squares_with_length
