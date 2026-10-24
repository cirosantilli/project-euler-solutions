import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12347_small : List Nat := []
theorem row12347_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12347 row12347_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12347_block_005 : List Nat := []
theorem row12347_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12347 row12347_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12347_block_006 : List Nat := []
def plan12347_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row12347_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12347 row12347_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12347_block_006
  · decide
  · rfl

def row12347_block_007 : List Nat := [13463]
def plan12347_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, .second, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 17), (.secondFactor 7), .possible, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12347_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12347 row12347_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12347_block_007
  · decide
  · rfl

def row12347_block_008 : List Nat := []
theorem row12347_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12347 row12347_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12347_block_009 : List Nat := []
theorem row12347_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12347 row12347_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12347_block_010 : List Nat := []
theorem row12347_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12347 row12347_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12347_block_011 : List Nat := []
theorem row12347_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12347 row12347_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12347_block_012 : List Nat := []
theorem row12347_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12347 row12347_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12347_block_013 : List Nat := []
theorem row12347_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12347 row12347_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12347_large_left_left : List Nat := row12347_small ++ row12347_block_005
theorem row12347_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12347 row12347_large_left_left :=
  Budget.RowUpperBound.branch row12347_small_covers row12347_block_005_covers

def row12347_large_left_right_right : List Nat := row12347_block_007 ++ row12347_block_008
theorem row12347_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12347 row12347_large_left_right_right :=
  Budget.RowUpperBound.branch row12347_block_007_covers row12347_block_008_covers

def row12347_large_left_right : List Nat := row12347_block_006 ++ row12347_large_left_right_right
theorem row12347_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12347 row12347_large_left_right :=
  Budget.RowUpperBound.branch row12347_block_006_covers row12347_large_left_right_right_covers

def row12347_large_left : List Nat := row12347_large_left_left ++ row12347_large_left_right
theorem row12347_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12347 row12347_large_left :=
  Budget.RowUpperBound.branch row12347_large_left_left_covers row12347_large_left_right_covers

def row12347_large_right_left : List Nat := row12347_block_009 ++ row12347_block_010
theorem row12347_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12347 row12347_large_right_left :=
  Budget.RowUpperBound.branch row12347_block_009_covers row12347_block_010_covers

def row12347_large_right_right_right : List Nat := row12347_block_012 ++ row12347_block_013
theorem row12347_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12347 row12347_large_right_right_right :=
  Budget.RowUpperBound.branch row12347_block_012_covers row12347_block_013_covers

def row12347_large_right_right : List Nat := row12347_block_011 ++ row12347_large_right_right_right
theorem row12347_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12347 row12347_large_right_right :=
  Budget.RowUpperBound.branch row12347_block_011_covers row12347_large_right_right_right_covers

def row12347_large_right : List Nat := row12347_large_right_left ++ row12347_large_right_right
theorem row12347_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12347 row12347_large_right :=
  Budget.RowUpperBound.branch row12347_large_right_left_covers row12347_large_right_right_covers

def row12347_large : List Nat := row12347_large_left ++ row12347_large_right
theorem row12347_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12347 row12347_large :=
  Budget.RowUpperBound.branch row12347_large_left_covers row12347_large_right_covers

def row12347 : List Nat := row12347_large
theorem row12347_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12347 row12347 := row12347_large_covers

def row12373_small : List Nat := []
theorem row12373_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12373 row12373_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12373_block_005 : List Nat := []
theorem row12373_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12373 row12373_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12373_block_006 : List Nat := []
def plan12373_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row12373_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12373 row12373_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12373_block_006
  · decide
  · rfl

def row12373_block_007 : List Nat := [13291]
def plan12373_block_007 : List Budget.Choice := [.second, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 37), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 13), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 23), .possible, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12373_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12373 row12373_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12373_block_007
  · decide
  · rfl

def row12373_block_008 : List Nat := []
theorem row12373_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12373 row12373_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12373_block_009 : List Nat := []
theorem row12373_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12373 row12373_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12373_block_010 : List Nat := []
theorem row12373_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12373 row12373_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12373_block_011 : List Nat := []
theorem row12373_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12373 row12373_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12373_block_012 : List Nat := []
theorem row12373_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12373 row12373_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12373_block_013 : List Nat := []
theorem row12373_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12373 row12373_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12373_large_left_left : List Nat := row12373_small ++ row12373_block_005
theorem row12373_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12373 row12373_large_left_left :=
  Budget.RowUpperBound.branch row12373_small_covers row12373_block_005_covers

def row12373_large_left_right_right : List Nat := row12373_block_007 ++ row12373_block_008
theorem row12373_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12373 row12373_large_left_right_right :=
  Budget.RowUpperBound.branch row12373_block_007_covers row12373_block_008_covers

def row12373_large_left_right : List Nat := row12373_block_006 ++ row12373_large_left_right_right
theorem row12373_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12373 row12373_large_left_right :=
  Budget.RowUpperBound.branch row12373_block_006_covers row12373_large_left_right_right_covers

def row12373_large_left : List Nat := row12373_large_left_left ++ row12373_large_left_right
theorem row12373_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12373 row12373_large_left :=
  Budget.RowUpperBound.branch row12373_large_left_left_covers row12373_large_left_right_covers

def row12373_large_right_left : List Nat := row12373_block_009 ++ row12373_block_010
theorem row12373_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12373 row12373_large_right_left :=
  Budget.RowUpperBound.branch row12373_block_009_covers row12373_block_010_covers

def row12373_large_right_right_right : List Nat := row12373_block_012 ++ row12373_block_013
theorem row12373_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12373 row12373_large_right_right_right :=
  Budget.RowUpperBound.branch row12373_block_012_covers row12373_block_013_covers

def row12373_large_right_right : List Nat := row12373_block_011 ++ row12373_large_right_right_right
theorem row12373_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12373 row12373_large_right_right :=
  Budget.RowUpperBound.branch row12373_block_011_covers row12373_large_right_right_right_covers

def row12373_large_right : List Nat := row12373_large_right_left ++ row12373_large_right_right
theorem row12373_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12373 row12373_large_right :=
  Budget.RowUpperBound.branch row12373_large_right_left_covers row12373_large_right_right_covers

def row12373_large : List Nat := row12373_large_left ++ row12373_large_right
theorem row12373_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12373 row12373_large :=
  Budget.RowUpperBound.branch row12373_large_left_covers row12373_large_right_covers

def row12373 : List Nat := row12373_large
theorem row12373_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12373 row12373 := row12373_large_covers

def row12377_small : List Nat := []
theorem row12377_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12377 row12377_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12377_block_005 : List Nat := []
theorem row12377_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12377 row12377_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12377_block_006 : List Nat := []
def plan12377_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row12377_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12377 row12377_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12377_block_006
  · decide
  · rfl

def row12377_block_007 : List Nat := [13457, 13469]
def plan12377_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .second, (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 19), (.firstFactor 7), .first, .second, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .second, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12377_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12377 row12377_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12377_block_007
  · decide
  · rfl

def row12377_block_008 : List Nat := []
theorem row12377_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12377 row12377_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12377_block_009 : List Nat := []
theorem row12377_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12377 row12377_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12377_block_010 : List Nat := []
theorem row12377_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12377 row12377_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12377_block_011 : List Nat := []
theorem row12377_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12377 row12377_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12377_block_012 : List Nat := []
theorem row12377_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12377 row12377_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12377_block_013 : List Nat := []
theorem row12377_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12377 row12377_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12377_large_left_left : List Nat := row12377_small ++ row12377_block_005
theorem row12377_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12377 row12377_large_left_left :=
  Budget.RowUpperBound.branch row12377_small_covers row12377_block_005_covers

def row12377_large_left_right_right : List Nat := row12377_block_007 ++ row12377_block_008
theorem row12377_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12377 row12377_large_left_right_right :=
  Budget.RowUpperBound.branch row12377_block_007_covers row12377_block_008_covers

def row12377_large_left_right : List Nat := row12377_block_006 ++ row12377_large_left_right_right
theorem row12377_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12377 row12377_large_left_right :=
  Budget.RowUpperBound.branch row12377_block_006_covers row12377_large_left_right_right_covers

def row12377_large_left : List Nat := row12377_large_left_left ++ row12377_large_left_right
theorem row12377_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12377 row12377_large_left :=
  Budget.RowUpperBound.branch row12377_large_left_left_covers row12377_large_left_right_covers

def row12377_large_right_left : List Nat := row12377_block_009 ++ row12377_block_010
theorem row12377_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12377 row12377_large_right_left :=
  Budget.RowUpperBound.branch row12377_block_009_covers row12377_block_010_covers

def row12377_large_right_right_right : List Nat := row12377_block_012 ++ row12377_block_013
theorem row12377_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12377 row12377_large_right_right_right :=
  Budget.RowUpperBound.branch row12377_block_012_covers row12377_block_013_covers

def row12377_large_right_right : List Nat := row12377_block_011 ++ row12377_large_right_right_right
theorem row12377_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12377 row12377_large_right_right :=
  Budget.RowUpperBound.branch row12377_block_011_covers row12377_large_right_right_right_covers

def row12377_large_right : List Nat := row12377_large_right_left ++ row12377_large_right_right
theorem row12377_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12377 row12377_large_right :=
  Budget.RowUpperBound.branch row12377_large_right_left_covers row12377_large_right_right_covers

def row12377_large : List Nat := row12377_large_left ++ row12377_large_right
theorem row12377_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12377 row12377_large :=
  Budget.RowUpperBound.branch row12377_large_left_covers row12377_large_right_covers

def row12377 : List Nat := row12377_large
theorem row12377_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12377 row12377 := row12377_large_covers

def row12379_small : List Nat := []
theorem row12379_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12379 row12379_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12379_block_005 : List Nat := []
theorem row12379_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12379 row12379_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12379_block_006 : List Nat := []
def plan12379_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 19), .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row12379_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12379 row12379_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12379_block_006
  · decide
  · rfl

def row12379_block_007 : List Nat := [13309, 13513]
def plan12379_block_007 : List Budget.Choice := [.first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 31), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .second, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12379_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12379 row12379_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12379_block_007
  · decide
  · rfl

def row12379_block_008 : List Nat := []
theorem row12379_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12379 row12379_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12379_block_009 : List Nat := []
theorem row12379_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12379 row12379_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12379_block_010 : List Nat := []
theorem row12379_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12379 row12379_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12379_block_011 : List Nat := []
theorem row12379_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12379 row12379_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12379_block_012 : List Nat := []
theorem row12379_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12379 row12379_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12379_block_013 : List Nat := []
theorem row12379_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12379 row12379_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12379_large_left_left : List Nat := row12379_small ++ row12379_block_005
theorem row12379_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12379 row12379_large_left_left :=
  Budget.RowUpperBound.branch row12379_small_covers row12379_block_005_covers

def row12379_large_left_right_right : List Nat := row12379_block_007 ++ row12379_block_008
theorem row12379_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12379 row12379_large_left_right_right :=
  Budget.RowUpperBound.branch row12379_block_007_covers row12379_block_008_covers

def row12379_large_left_right : List Nat := row12379_block_006 ++ row12379_large_left_right_right
theorem row12379_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12379 row12379_large_left_right :=
  Budget.RowUpperBound.branch row12379_block_006_covers row12379_large_left_right_right_covers

def row12379_large_left : List Nat := row12379_large_left_left ++ row12379_large_left_right
theorem row12379_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12379 row12379_large_left :=
  Budget.RowUpperBound.branch row12379_large_left_left_covers row12379_large_left_right_covers

def row12379_large_right_left : List Nat := row12379_block_009 ++ row12379_block_010
theorem row12379_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12379 row12379_large_right_left :=
  Budget.RowUpperBound.branch row12379_block_009_covers row12379_block_010_covers

def row12379_large_right_right_right : List Nat := row12379_block_012 ++ row12379_block_013
theorem row12379_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12379 row12379_large_right_right_right :=
  Budget.RowUpperBound.branch row12379_block_012_covers row12379_block_013_covers

def row12379_large_right_right : List Nat := row12379_block_011 ++ row12379_large_right_right_right
theorem row12379_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12379 row12379_large_right_right :=
  Budget.RowUpperBound.branch row12379_block_011_covers row12379_large_right_right_right_covers

def row12379_large_right : List Nat := row12379_large_right_left ++ row12379_large_right_right
theorem row12379_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12379 row12379_large_right :=
  Budget.RowUpperBound.branch row12379_large_right_left_covers row12379_large_right_right_covers

def row12379_large : List Nat := row12379_large_left ++ row12379_large_right
theorem row12379_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12379 row12379_large :=
  Budget.RowUpperBound.branch row12379_large_left_covers row12379_large_right_covers

def row12379 : List Nat := row12379_large
theorem row12379_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12379 row12379 := row12379_large_covers

end ProjectEulerSolutions.PrimePairLarge
