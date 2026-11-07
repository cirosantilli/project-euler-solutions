import ProjectEulerSolutions.Lib.PatternBuckets

namespace ProjectEulerSolutions.PatternBuckets
open FormalSW.PyAst HeapCalls
set_option Elab.async false

theorem get (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (found : heap[dictionary]? = some (object buckets)) (key : Key) (representative : Nat)
    (keyFound : heap[representative]? = some (.tuple (key.map number))) :
    Mutable.dictionaryGet dictionary (.ref representative) (.ref heap.length) (prepared heap) =
      some (.ok (.ref (chosen buckets key heap.length)), prepared heap) := by
  have original : (prepared heap)[dictionary]? = some (object buckets) :=
    HeapCollection.prefix_found heap [.list []] dictionary _ found
  have hashed : Mutable.hashKey (.ref representative) (prepared heap) = some (.ok (hash key), prepared heap) := by
    have kept := HeapCollection.prefix_found heap [.list []] representative _ keyFound
    simpa [hash, number, List.map_map, Function.comp_def] using
      Mutable.hashKey_tuple_scalars (prepared heap) representative (key.map fun (n : Nat) => Constant.int n)
        (by simpa [prepared, number, List.map_map, Function.comp_def] using kept)
  cases buckets with
  | nil => simp [Mutable.dictionaryGet, original, object, hashed, Mutable.promoteDictionary, chosen, find, Except.map]
  | cons first rest =>
      have operation := Mutable.dictionaryGet_keyed (prepared heap) dictionary (entries (first :: rest)) original
        (.ref representative) (hash key) hashed (.ref heap.length)
      cases selected : find (first :: rest) key <;>
        simpa only [lookup_entries, selected, Option.map_none, Option.map_some, Option.getD_none, Option.getD_some, chosen] using operation

theorem get_values (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) :
    (prepared heap)[chosen buckets key heap.length]? = some (.list (texts (contents buckets key))) := by
  cases selected : find buckets key with
  | none => simp [chosen, contents, selected, prepared, texts]
  | some bucket =>
      simpa only [chosen, contents, selected, Option.map_some, Option.getD_some] using
        HeapCollection.prefix_found heap [.list []] bucket.address _ (view.lists bucket (List.mem_of_find?_eq_some selected))

/-- Bind the actual dictionary receiver before allocating the eager empty-list default. -/
theorem evaluate_get {globals : Mutable.Globals} (program : Module) (env : HeapCalls.Env) (receiver : Expr)
    (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (view : View heap dictionary buckets)
    (key : Key) (representative : Nat) (localName : ∀ name, receiver = .name name → env name ≠ none)
    (evaluated : Evaluates program env receiver heap (.ref dictionary) heap globals)
    (getKey : env "p1" = some (.ref representative)) (keyFound : heap[representative]? = some (.tuple (key.map number))) :
    Evaluates program env (.call (.attribute receiver "get") [.name "p1", .list []]) heap
      (.ref (chosen buckets key heap.length)) (prepared heap) globals := by
  have bound : bindMethod program (.ref dictionary) "get" heap = some (.ok (.dictionaryGet dictionary), heap) := by
    have found := view.found
    cases buckets <;> simp_all [bindMethod, object]
  apply Evaluates.dictionary_get_bound localName evaluated bound (.cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil))
  exact get heap dictionary buckets view.found key representative keyFound

end ProjectEulerSolutions.PatternBuckets
