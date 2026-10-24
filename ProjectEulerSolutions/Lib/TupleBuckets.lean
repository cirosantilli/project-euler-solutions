import lang.py.FormalSW.KeyDictionaryCalls
import ProjectEulerSolutions.Lib.HeapCollection
import Mathlib.Data.List.Nodup

/-! Tuple-key dictionaries whose values are distinct, shared lists of natural numbers. -/
namespace ProjectEulerSolutions.TupleBuckets
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

abbrev Key := Nat × Nat
def number (n : Nat) : Mutable.Value := .scalar (.int n)
def numbers (items : List Nat) : List Mutable.Value := items.map number
def hash (key : Key) : HashKey := .tuple [.scalar (.int key.1), .scalar (.int key.2)]

structure Bucket where
  key : Key
  representative : Nat
  address : Nat
  values : List Nat
deriving Repr, DecidableEq

def entry (bucket : Bucket) : HashDict.Entry Mutable.Value :=
  (hash bucket.key, .ref bucket.representative, .ref bucket.address)
def entries (buckets : List Bucket) := buckets.map entry
def object : List Bucket → Mutable.Object
  | [] => .dict []
  | first :: rest => .keyDict (entries (first :: rest))
def find (buckets : List Bucket) (key : Key) : Option Bucket :=
  buckets.find? (fun bucket => decide (bucket.key = key))
def chosen (buckets : List Bucket) (key : Key) (fresh : Nat) : Nat :=
  ((find buckets key).map Bucket.address).getD fresh
def contents (buckets : List Bucket) (key : Key) : List Nat :=
  ((find buckets key).map Bucket.values).getD []

/-- Existing keys keep both object identities; a missing key receives the fresh default list. -/
def push : List Bucket → Key → Nat → Nat → Nat → List Bucket
  | [], key, representative, address, value => [⟨key, representative, address, [value]⟩]
  | bucket :: rest, key, representative, address, value =>
      if bucket.key = key then { bucket with values := bucket.values ++ [value] } :: rest
      else bucket :: push rest key representative address value

theorem hash_equal (left right : Key) : HashKey.equal (hash left) (hash right) = decide (left = right) := by
  rcases left with ⟨a,b⟩
  rcases right with ⟨c,d⟩
  simp [hash, HashKey.equal, HashKey.depth, HashKey.listDepth, HashKey.equalFuel,
    List.isEqv, scalarCompare, evalCmpOp, Prod.ext_iff, Int.natCast_inj]

theorem lookup_entries (buckets : List Bucket) (key : Key) :
    HashDict.lookup (entries buckets) (hash key) = (find buckets key).map (fun bucket => .ref bucket.address) := by
  induction buckets with
  | nil => rfl
  | cons bucket rest ih =>
      simp only [entries, List.map_cons, HashDict.lookup_cons, entry, hash_equal]
      by_cases same : bucket.key = key
      · simp [find, same]
      · simpa only [find, List.find?_cons, show decide (bucket.key = key) = false by simp [same],
          if_false] using ih

theorem entries_push (buckets : List Bucket) (key : Key) (representative address value : Nat) :
    entries (push buckets key representative address value) =
      HashDict.insert (entries buckets) (hash key) (.ref representative) (.ref (chosen buckets key address)) := by
  induction buckets with
  | nil => rfl
  | cons bucket rest ih =>
      by_cases same : bucket.key = key
      · simp [push, same, entries, entry, HashDict.insert, hash_equal, chosen, find]
      · simpa [push, same, entries, entry, HashDict.insert, hash_equal, chosen, find] using ih

theorem push_nonempty (buckets : List Bucket) (key : Key) (representative address value : Nat) :
    push buckets key representative address value ≠ [] := by
  cases buckets with
  | nil => simp [push]
  | cons bucket rest => by_cases same : bucket.key = key <;> simp [push, same]

theorem object_push (buckets : List Bucket) (key : Key) (representative address value : Nat) :
    object (push buckets key representative address value) =
      .keyDict (entries (push buckets key representative address value)) := by
  cases changed : push buckets key representative address value with
  | nil => exact (push_nonempty buckets key representative address value changed).elim
  | cons first rest => rfl

theorem entries_push_hit (buckets : List Bucket) (key : Key) (representative address value : Nat)
    (bucket : Bucket) (found : find buckets key = some bucket) :
    entries (push buckets key representative address value) = entries buckets := by
  induction buckets with
  | nil => simp [find] at found
  | cons first rest ih =>
      by_cases same : first.key = key
      · simp [push, same, entries, entry]
      · have tail : find rest key = some bucket := by simpa [find, same] using found
        simpa only [push, if_neg same, entries, List.map_cons] using congrArg (List.cons (entry first)) (ih tail)

theorem contents_push (buckets : List Bucket) (key query : Key) (representative address value : Nat) :
    contents (push buckets key representative address value) query =
      if query = key then contents buckets key ++ [value] else contents buckets query := by
  induction buckets with
  | nil => by_cases same : key = query <;> simp [push, contents, find, same, Ne.symm]
  | cons first rest ih =>
      by_cases updated : first.key = key <;> by_cases selected : first.key = query <;>
        by_cases same : query = key <;> simp_all [push, contents, find]

theorem push_miss (buckets : List Bucket) (key : Key) (representative address value : Nat)
    (missing : find buckets key = none) :
    push buckets key representative address value = buckets ++ [⟨key, representative, address, [value]⟩] := by
  induction buckets with
  | nil => rfl
  | cons first rest ih =>
      have different : first.key ≠ key := by
        intro same
        simp [find, same] at missing
      have tail : find rest key = none := by simpa [find, different] using missing
      simp [push, different, ih tail]

theorem push_split (before after : List Bucket) (selected : Bucket) (key : Key)
    (matched : selected.key = key) (skipped : ∀ bucket ∈ before, bucket.key ≠ key)
    (representative address value : Nat) :
    push (before ++ selected :: after) key representative address value =
      before ++ { selected with values := selected.values ++ [value] } :: after := by
  induction before with
  | nil => simp [push, matched]
  | cons first rest ih =>
      have different := skipped first List.mem_cons_self
      simp only [List.cons_append, push, if_neg different]
      rw [ih (fun bucket member => skipped bucket (List.mem_cons_of_mem _ member))]

theorem addresses_push (buckets : List Bucket) (key : Key) (representative address value : Nat) :
    (push buckets key representative address value).map Bucket.address =
      if (find buckets key).isSome then buckets.map Bucket.address else buckets.map Bucket.address ++ [address] := by
  induction buckets with
  | nil => rfl
  | cons first rest ih =>
      by_cases same : first.key = key
      · simp [push, same, find]
      · have skipped : find (first :: rest) key = find rest key := by simp [find, same]
        simp only [push, if_neg same, List.map_cons, ih, skipped]
        split <;> simp_all

structure View (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) : Prop where
  found : heap[dictionary]? = some (object buckets)
  lists : ∀ bucket ∈ buckets, heap[bucket.address]? = some (.list (numbers bucket.values))
  afterDictionary : ∀ bucket ∈ buckets, dictionary < bucket.address
  unique : (buckets.map Bucket.address).Nodup

theorem View.append {heap : Mutable.Heap} {dictionary : Nat} {buckets : List Bucket}
    (view : View heap dictionary buckets) (extra : Mutable.Heap) : View (heap ++ extra) dictionary buckets :=
  ⟨HeapCollection.prefix_found heap extra dictionary _ view.found,
    fun bucket member => HeapCollection.prefix_found heap extra bucket.address _ (view.lists bucket member),
    view.afterDictionary, view.unique⟩

theorem empty_view (heap : Mutable.Heap) : View (heap ++ [.dict []]) heap.length [] := by
  constructor
  · simp [object]
  · simp
  · simp
  · simp

def prepared (heap : Mutable.Heap) (key : Key) : Mutable.Heap :=
  heap ++ [.tuple [number key.1, number key.2], .list []]

def pending (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (key : Key) (value : Nat) : Mutable.Heap :=
  if (find buckets key).isSome then prepared heap key
  else (prepared heap key).set dictionary (object (push buckets key heap.length (heap.length+1) value))

/-- Every call evaluates and allocates its tuple key and default before the actual dictionary lookup. -/
theorem setdefault (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (found : heap[dictionary]? = some (object buckets)) (key : Key) (value : Nat) :
    Mutable.dictionarySetDefault dictionary (.ref heap.length) (.ref (heap.length+1)) (prepared heap key) =
      some (.ok (.ref (chosen buckets key (heap.length+1))), pending heap dictionary buckets key value) := by
  have original : (prepared heap key)[dictionary]? = some (object buckets) :=
    HeapCollection.prefix_found heap _ dictionary _ found
  have hashed : Mutable.hashKey (.ref heap.length) (prepared heap key) = some (.ok (hash key), prepared heap key) := by
    exact Mutable.hashKey_tuple_scalars (prepared heap key) heap.length [.int key.1, .int key.2]
      (by simp [prepared, number])
  cases buckets with
  | nil =>
      have operation := Mutable.dictionarySetDefault_promote (prepared heap key) dictionary [] original
        heap.length (hash key) hashed (.ref (heap.length+1))
      simpa [pending, find, chosen, object, push, entries, entry, Mutable.promoteDictionary] using operation
  | cons first rest =>
      have operation := Mutable.dictionarySetDefault_keyed (prepared heap key) dictionary
        (entries (first :: rest)) original (.ref heap.length) (hash key) hashed (.ref (heap.length+1))
      cases selected : find (first :: rest) key with
      | none =>
          simpa only [lookup_entries, selected, Option.map_none, Option.getD_none, Option.isSome_none,
            Bool.false_eq_true, if_false, pending, chosen, object_push, entries_push] using operation
      | some bucket =>
          simpa only [lookup_entries, selected, Option.map_some, Option.getD_some, Option.isSome_some,
            if_true, pending, chosen] using operation

theorem selected_values (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (value : Nat) :
    (pending heap dictionary buckets key value)[chosen buckets key (heap.length+1)]? =
      some (.list (numbers (contents buckets key))) := by
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      have different : heap.length+1 ≠ dictionary := by omega
      simp only [pending, chosen, contents, selected, Option.isSome_none, Bool.false_eq_true,
        if_false, Option.map_none, Option.getD_none]
      rw [List.getElem?_set_ne (Ne.symm different)]
      simp [prepared, numbers]
  | some bucket =>
      have member : bucket ∈ buckets := List.mem_of_find?_eq_some selected
      have found := view.lists bucket member
      simpa [pending, chosen, contents, selected] using
        HeapCollection.prefix_found heap [.tuple [number key.1, number key.2], .list []] bucket.address _ found

def updated (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (key : Key) (value : Nat) : Mutable.Heap :=
  (pending heap dictionary buckets key value).set (chosen buckets key (heap.length+1))
    (.list (numbers (contents buckets key ++ [value])))

theorem chosen_after (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) : dictionary < chosen buckets key (heap.length+1) := by
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      simp only [chosen, selected, Option.map_none, Option.getD_none]
      omega
  | some bucket =>
      simpa only [chosen, selected, Option.map_some, Option.getD_some] using
        view.afterDictionary bucket (List.mem_of_find?_eq_some selected)

theorem updated_selected (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (value : Nat) :
    (updated heap dictionary buckets key value)[chosen buckets key (heap.length+1)]? =
      some (.list (numbers (contents buckets key ++ [value]))) := by
  exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp (selected_values heap dictionary buckets view key value)).1

theorem updated_other (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (value : Nat) (bucket : Bucket)
    (member : bucket ∈ buckets) (different : chosen buckets key (heap.length+1) ≠ bucket.address) :
    (updated heap dictionary buckets key value)[bucket.address]? = heap[bucket.address]? := by
  unfold updated
  rw [List.getElem?_set_ne different]
  have inside := (List.getElem?_eq_some_iff.mp (view.lists bucket member)).1
  have afterDictionary := view.afterDictionary bucket member
  unfold pending
  split
  · exact List.getElem?_append_left inside
  · rw [List.getElem?_set_ne (by omega : dictionary ≠ bucket.address)]
    exact List.getElem?_append_left inside

theorem updated_dictionary (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (value : Nat) :
    (updated heap dictionary buckets key value)[dictionary]? =
      some (object (push buckets key heap.length (heap.length+1) value)) := by
  have different := chosen_after heap dictionary buckets view key
  unfold updated
  rw [List.getElem?_set_ne (by omega : chosen buckets key (heap.length+1) ≠ dictionary)]
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      simp only [pending, selected, Option.isSome_none, Bool.false_eq_true, if_false]
      exact List.getElem?_set_self (by simp [prepared]; omega)
  | some bucket =>
      have same : object (push buckets key heap.length (heap.length+1) value) = object buckets := by
        rw [object_push, entries_push_hit buckets key _ _ _ bucket selected]
        cases buckets with
        | nil => simp [find] at selected
        | cons first rest => rfl
      rw [same]
      simpa only [pending, selected, Option.isSome_some, if_true] using
        HeapCollection.prefix_found heap _ dictionary _ view.found

/-- All writes stay at or above the dictionary, so a preceding input heap remains unchanged. -/
theorem updated_prefix (base suffix : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View (base ++ suffix) dictionary buckets) (afterBase : base.length ≤ dictionary)
    (key : Key) (value : Nat) :
    ∃ extra, updated (base ++ suffix) dictionary buckets key value = base ++ extra := by
  have afterChosen : base.length ≤ chosen buckets key ((base ++ suffix).length+1) := by
    have later := chosen_after (base ++ suffix) dictionary buckets view key
    omega
  have preparedPrefix : prepared (base ++ suffix) key =
      base ++ (suffix ++ [.tuple [number key.1, number key.2], .list []]) := by simp [prepared, List.append_assoc]
  unfold updated pending
  split
  · rw [preparedPrefix, List.set_append_right _ _ afterChosen]
    exact ⟨_, rfl⟩
  · rw [preparedPrefix, List.set_append_right _ _ afterBase, List.set_append_right _ _ afterChosen]
    exact ⟨_, rfl⟩

/-- Appending to one bucket preserves the dictionary, every other list, and absence of aliases. -/
theorem updated_view (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (value : Nat) :
    View (updated heap dictionary buckets key value) dictionary
      (push buckets key heap.length (heap.length+1) value) := by
  have fresh : heap.length+1 ∉ buckets.map Bucket.address := by
    intro member
    obtain ⟨bucket, inside, equal⟩ := List.mem_map.mp member
    have bound := (List.getElem?_eq_some_iff.mp (view.lists bucket inside)).1
    omega
  constructor
  · exact updated_dictionary heap dictionary buckets view key value
  · intro current member
    cases selected : find buckets key with
    | none =>
        rw [push_miss buckets key _ _ _ selected] at member
        rcases List.mem_append.mp member with old | added
        · have bound := (List.getElem?_eq_some_iff.mp (view.lists current old)).1
          have different : chosen buckets key (heap.length+1) ≠ current.address := by
            simp only [chosen, selected, Option.map_none, Option.getD_none]
            omega
          rw [updated_other heap dictionary buckets view key value current old different]
          exact view.lists current old
        · obtain rfl := List.mem_singleton.mp added
          simpa only [chosen, contents, selected, Option.map_none, Option.getD_none, List.nil_append] using
            updated_selected heap dictionary buckets view key value
    | some bucket =>
        obtain ⟨matched, before, after, split, skipped⟩ := List.find?_eq_some_iff_append.mp selected
        have keyEqual : bucket.key = key := of_decide_eq_true matched
        have skip : ∀ entry ∈ before, entry.key ≠ key := by
          intro entry inside
          simpa using skipped entry inside
        have distinct : (before.map Bucket.address ++ bucket.address :: after.map Bucket.address).Nodup := by
          simpa only [split, List.map_append, List.map_cons] using view.unique
        have moved : (bucket.address :: (before.map Bucket.address ++ after.map Bucket.address)).Nodup :=
          List.perm_middle.nodup_iff.mp distinct
        have without := (List.nodup_cons.mp moved).1
        have kept (entry : Bucket) (inside : entry ∈ before ++ after) :
            (updated heap dictionary buckets key value)[entry.address]? = some (.list (numbers entry.values)) := by
          have old : entry ∈ buckets := by
            rw [split]
            rcases List.mem_append.mp inside with left | right
            · exact List.mem_append.mpr (.inl left)
            · exact List.mem_append.mpr (.inr (List.mem_cons_of_mem _ right))
          have different : chosen buckets key (heap.length+1) ≠ entry.address := by
            simp only [chosen, selected, Option.map_some, Option.getD_some]
            intro equal
            apply without
            rw [equal]
            simpa only [List.map_append] using List.mem_map_of_mem (f := Bucket.address) inside
          rw [updated_other heap dictionary buckets view key value entry old different]
          exact view.lists entry old
        rw [split, push_split before after bucket key keyEqual skip] at member
        rcases List.mem_append.mp member with left | right
        · exact kept current (List.mem_append.mpr (.inl left))
        · rcases List.mem_cons.mp right with rfl | right
          · simpa only [chosen, contents, selected, Option.map_some, Option.getD_some] using
              updated_selected heap dictionary buckets view key value
          · exact kept current (List.mem_append.mpr (.inr right))
  · intro current member
    have addressMember := List.mem_map_of_mem (f := Bucket.address) member
    rw [addresses_push] at addressMember
    cases selected : find buckets key with
    | none =>
        simp only [selected, Option.isSome_none, Bool.false_eq_true, if_false] at addressMember
        rcases List.mem_append.mp addressMember with old | added
        · obtain ⟨entry, inside, equal⟩ := List.mem_map.mp old
          rw [← equal]
          exact view.afterDictionary entry inside
        · have equal := List.mem_singleton.mp added
          have bound := (List.getElem?_eq_some_iff.mp view.found).1
          omega
    | some bucket =>
        simp only [selected, Option.isSome_some, if_true] at addressMember
        obtain ⟨entry, inside, equal⟩ := List.mem_map.mp addressMember
        rw [← equal]
        exact view.afterDictionary entry inside
  · rw [addresses_push]
    split
    · exact view.unique
    · simpa only [List.concat_eq_append] using List.Nodup.concat fresh view.unique

/-- Read-only lookup retains both eagerly allocated arguments, even when the key already exists. -/
theorem get (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (found : heap[dictionary]? = some (object buckets)) (key : Key) :
    Mutable.dictionaryGet dictionary (.ref heap.length) (.ref (heap.length+1)) (prepared heap key) =
      some (.ok (.ref (chosen buckets key (heap.length+1))), prepared heap key) := by
  have original : (prepared heap key)[dictionary]? = some (object buckets) :=
    HeapCollection.prefix_found heap _ dictionary _ found
  have hashed : Mutable.hashKey (.ref heap.length) (prepared heap key) = some (.ok (hash key), prepared heap key) :=
    Mutable.hashKey_tuple_scalars (prepared heap key) heap.length [.int key.1, .int key.2]
      (by simp [prepared, number])
  cases buckets with
  | nil => simp [Mutable.dictionaryGet, original, object, hashed, Mutable.promoteDictionary, chosen, find, Except.map]
  | cons first rest =>
      have operation := Mutable.dictionaryGet_keyed (prepared heap key) dictionary (entries (first :: rest))
        original (.ref heap.length) (hash key) hashed (.ref (heap.length+1))
      cases selected : find (first :: rest) key <;>
        simpa only [lookup_entries, selected, Option.map_none, Option.map_some,
          Option.getD_none, Option.getD_some, chosen] using operation

/-- A missing key reads the freshly allocated empty default; existing lists keep their identity. -/
theorem get_values (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) :
    (prepared heap key)[chosen buckets key (heap.length+1)]? = some (.list (numbers (contents buckets key))) := by
  cases selected : find buckets key with
  | none => simp [chosen, contents, selected, prepared, numbers]
  | some bucket =>
      simpa only [chosen, contents, selected, Option.map_some, Option.getD_some] using
        HeapCollection.prefix_found heap _ bucket.address _ (view.lists bucket (List.mem_of_find?_eq_some selected))

def lookupExpression : Expr := .call (.attribute (.name "mp") "get") [.tuple [.name "b", .name "c"], .list []]

/-- Execute the source lookup, preserving the shared bucket list and the complete original heap. -/
theorem evaluate_get {globals : Mutable.Globals} (program : Module) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (view : View heap dictionary buckets)
    (key : Key) (getMap : env "mp" = some (.ref dictionary))
    (getB : env "b" = some (number key.1)) (getC : env "c" = some (number key.2)) :
    Evaluates (globals := globals) program env lookupExpression heap
      (.ref (chosen buckets key (heap.length+1))) (prepared heap key) := by
  have keyResult : Evaluates (globals := globals) program env (.tuple [.name "b", .name "c"])
      heap (.ref heap.length) (heap ++ [.tuple [number key.1, number key.2]]) :=
    Evaluates.tuple (.cons (Evaluates.lookup getB) (.cons (Evaluates.lookup getC) .nil)) (by simp)
  have defaultResult : Evaluates (globals := globals) program env (.list [])
      (heap ++ [.tuple [number key.1, number key.2]]) (.ref (heap.length+1)) (prepared heap key) := by
    have allocated : Evaluates (globals := globals) program env (.list [])
        (heap ++ [.tuple [number key.1, number key.2]])
        (.ref ((heap ++ [Mutable.Object.tuple [number key.1, number key.2]]).length))
        ((heap ++ [.tuple [number key.1, number key.2]]) ++ [.list []]) := Evaluates.list .nil
    simpa only [prepared, List.length_append, List.length_singleton, List.append_assoc, List.singleton_append] using allocated
  have bound : bindMethod program (.ref dictionary) "get" heap = some (.ok (.dictionaryGet dictionary), heap) := by
    have found := view.found
    cases buckets <;> simp_all [bindMethod, object]
  apply Evaluates.dictionary_get_bound (by intro name same; cases same; simp [getMap])
    (Evaluates.lookup getMap) bound (.cons keyResult (.cons defaultResult .nil))
  exact get heap dictionary buckets view.found key

def insertion : Expr := .call
  (.attribute (.call (.attribute (.name "mp") "setdefault")
    [.tuple [.name "b", .name "c"], .list []]) "append") [.name "a"]

/-- This is the source expression, including receiver capture and both eager argument allocations. -/
theorem evaluate {globals : Mutable.Globals} (program : Module) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (view : View heap dictionary buckets)
    (key : Key) (value : Nat) (getMap : env "mp" = some (.ref dictionary))
    (getA : env "a" = some (number value)) (getB : env "b" = some (number key.1))
    (getC : env "c" = some (number key.2)) :
    Evaluates (globals := globals) program env insertion heap (.scalar .none)
      (updated heap dictionary buckets key value) := by
  have keyResult : Evaluates (globals := globals) program env (.tuple [.name "b", .name "c"])
      heap (.ref heap.length) (heap ++ [.tuple [number key.1, number key.2]]) :=
    Evaluates.tuple (.cons (Evaluates.lookup getB) (.cons (Evaluates.lookup getC) .nil)) (by simp)
  have defaultResult : Evaluates (globals := globals) program env (.list [])
      (heap ++ [.tuple [number key.1, number key.2]]) (.ref (heap.length+1)) (prepared heap key) := by
    have allocated : Evaluates (globals := globals) program env (.list [])
        (heap ++ [.tuple [number key.1, number key.2]])
        (.ref ((heap ++ [Mutable.Object.tuple [number key.1, number key.2]]).length))
        ((heap ++ [.tuple [number key.1, number key.2]]) ++ [.list []]) := Evaluates.list .nil
    simpa only [prepared, List.length_append, List.length_singleton, List.append_assoc, List.singleton_append] using
      allocated
  have bound : bindMethod program (.ref dictionary) "setdefault" heap =
      some (.ok (.dictionarySetDefault dictionary), heap) := by
    have found := view.found
    cases buckets <;> simp_all [bindMethod, object]
  have receiver : Evaluates (globals := globals) program env
      (.call (.attribute (.name "mp") "setdefault") [.tuple [.name "b", .name "c"], .list []])
      heap (.ref (chosen buckets key (heap.length+1))) (pending heap dictionary buckets key value) := by
    apply Evaluates.dictionary_setdefault
      (by intro name same; cases same; simp [getMap]) (Evaluates.lookup getMap) bound
      (.cons keyResult (.cons defaultResult .nil))
    exact setdefault heap dictionary buckets view.found key value
  have selected := selected_values heap dictionary buckets view key value
  simpa only [insertion, updated, numbers, List.map_append, List.map_cons, List.map_nil] using
    Evaluates.append receiver selected (Evaluates.lookup getA) selected

end ProjectEulerSolutions.TupleBuckets
