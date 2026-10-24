import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12269_small : List Nat := []
theorem row12269_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12269 row12269_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12269_block_005 : List Nat := []
theorem row12269_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12269 row12269_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12269_block_006 : List Nat := []
def plan12269_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row12269_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12269 row12269_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12269_block_006
  · decide
  · rfl

def row12269_block_007 : List Nat := [12671]
def plan12269_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 23), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 13), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 23), .first, .second, (.secondFactor 29), (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 29), (.firstFactor 29), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12269_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12269 row12269_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12269_block_007
  · decide
  · rfl

def row12269_block_008 : List Nat := []
theorem row12269_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12269 row12269_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12269_block_009 : List Nat := []
theorem row12269_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12269 row12269_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12269_block_010 : List Nat := []
theorem row12269_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12269 row12269_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12269_block_011 : List Nat := []
theorem row12269_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12269 row12269_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12269_block_012 : List Nat := []
theorem row12269_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12269 row12269_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12269_block_013 : List Nat := []
theorem row12269_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12269 row12269_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12269_large_left_left : List Nat := row12269_small ++ row12269_block_005
theorem row12269_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12269 row12269_large_left_left :=
  Budget.RowUpperBound.branch row12269_small_covers row12269_block_005_covers

def row12269_large_left_right_right : List Nat := row12269_block_007 ++ row12269_block_008
theorem row12269_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12269 row12269_large_left_right_right :=
  Budget.RowUpperBound.branch row12269_block_007_covers row12269_block_008_covers

def row12269_large_left_right : List Nat := row12269_block_006 ++ row12269_large_left_right_right
theorem row12269_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12269 row12269_large_left_right :=
  Budget.RowUpperBound.branch row12269_block_006_covers row12269_large_left_right_right_covers

def row12269_large_left : List Nat := row12269_large_left_left ++ row12269_large_left_right
theorem row12269_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12269 row12269_large_left :=
  Budget.RowUpperBound.branch row12269_large_left_left_covers row12269_large_left_right_covers

def row12269_large_right_left : List Nat := row12269_block_009 ++ row12269_block_010
theorem row12269_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12269 row12269_large_right_left :=
  Budget.RowUpperBound.branch row12269_block_009_covers row12269_block_010_covers

def row12269_large_right_right_right : List Nat := row12269_block_012 ++ row12269_block_013
theorem row12269_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12269 row12269_large_right_right_right :=
  Budget.RowUpperBound.branch row12269_block_012_covers row12269_block_013_covers

def row12269_large_right_right : List Nat := row12269_block_011 ++ row12269_large_right_right_right
theorem row12269_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12269 row12269_large_right_right :=
  Budget.RowUpperBound.branch row12269_block_011_covers row12269_large_right_right_right_covers

def row12269_large_right : List Nat := row12269_large_right_left ++ row12269_large_right_right
theorem row12269_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12269 row12269_large_right :=
  Budget.RowUpperBound.branch row12269_large_right_left_covers row12269_large_right_right_covers

def row12269_large : List Nat := row12269_large_left ++ row12269_large_right
theorem row12269_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12269 row12269_large :=
  Budget.RowUpperBound.branch row12269_large_left_covers row12269_large_right_covers

def row12269 : List Nat := row12269_large
theorem row12269_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12269 row12269 := row12269_large_covers

def row12277_small : List Nat := []
theorem row12277_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12277 row12277_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12277_block_005 : List Nat := []
theorem row12277_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12277 row12277_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12277_block_006 : List Nat := [12373]
def plan12277_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row12277_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12277 row12277_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12277_block_006
  · decide
  · rfl

def row12277_block_007 : List Nat := [12637, 13411]
def plan12277_block_007 : List Budget.Choice := [(.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 7), .second, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .second, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12277_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12277 row12277_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12277_block_007
  · decide
  · rfl

def row12277_block_008 : List Nat := []
theorem row12277_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12277 row12277_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12277_block_009 : List Nat := []
theorem row12277_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12277 row12277_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12277_block_010 : List Nat := []
theorem row12277_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12277 row12277_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12277_block_011 : List Nat := []
theorem row12277_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12277 row12277_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12277_block_012 : List Nat := []
theorem row12277_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12277 row12277_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12277_block_013 : List Nat := []
theorem row12277_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12277 row12277_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12277_large_left_left : List Nat := row12277_small ++ row12277_block_005
theorem row12277_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12277 row12277_large_left_left :=
  Budget.RowUpperBound.branch row12277_small_covers row12277_block_005_covers

def row12277_large_left_right_right : List Nat := row12277_block_007 ++ row12277_block_008
theorem row12277_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12277 row12277_large_left_right_right :=
  Budget.RowUpperBound.branch row12277_block_007_covers row12277_block_008_covers

def row12277_large_left_right : List Nat := row12277_block_006 ++ row12277_large_left_right_right
theorem row12277_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12277 row12277_large_left_right :=
  Budget.RowUpperBound.branch row12277_block_006_covers row12277_large_left_right_right_covers

def row12277_large_left : List Nat := row12277_large_left_left ++ row12277_large_left_right
theorem row12277_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12277 row12277_large_left :=
  Budget.RowUpperBound.branch row12277_large_left_left_covers row12277_large_left_right_covers

def row12277_large_right_left : List Nat := row12277_block_009 ++ row12277_block_010
theorem row12277_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12277 row12277_large_right_left :=
  Budget.RowUpperBound.branch row12277_block_009_covers row12277_block_010_covers

def row12277_large_right_right_right : List Nat := row12277_block_012 ++ row12277_block_013
theorem row12277_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12277 row12277_large_right_right_right :=
  Budget.RowUpperBound.branch row12277_block_012_covers row12277_block_013_covers

def row12277_large_right_right : List Nat := row12277_block_011 ++ row12277_large_right_right_right
theorem row12277_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12277 row12277_large_right_right :=
  Budget.RowUpperBound.branch row12277_block_011_covers row12277_large_right_right_right_covers

def row12277_large_right : List Nat := row12277_large_right_left ++ row12277_large_right_right
theorem row12277_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12277 row12277_large_right :=
  Budget.RowUpperBound.branch row12277_large_right_left_covers row12277_large_right_right_covers

def row12277_large : List Nat := row12277_large_left ++ row12277_large_right
theorem row12277_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12277 row12277_large :=
  Budget.RowUpperBound.branch row12277_large_left_covers row12277_large_right_covers

def row12277 : List Nat := row12277_large
theorem row12277_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12277 row12277 := row12277_large_covers

def row12281_small : List Nat := []
theorem row12281_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12281 row12281_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12281_block_005 : List Nat := []
theorem row12281_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12281 row12281_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12281_block_006 : List Nat := []
def plan12281_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row12281_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12281 row12281_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12281_block_006
  · decide
  · rfl

def row12281_block_007 : List Nat := [13121, 13499]
def plan12281_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .second, (.firstFactor 29), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .second, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, (.firstFactor 37), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .second, .first, (.firstFactor 29), .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12281_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12281 row12281_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12281_block_007
  · decide
  · rfl

def row12281_block_008 : List Nat := []
theorem row12281_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12281 row12281_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12281_block_009 : List Nat := []
theorem row12281_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12281 row12281_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12281_block_010 : List Nat := []
theorem row12281_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12281 row12281_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12281_block_011 : List Nat := []
theorem row12281_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12281 row12281_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12281_block_012 : List Nat := []
theorem row12281_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12281 row12281_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12281_block_013 : List Nat := []
theorem row12281_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12281 row12281_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12281_large_left_left : List Nat := row12281_small ++ row12281_block_005
theorem row12281_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12281 row12281_large_left_left :=
  Budget.RowUpperBound.branch row12281_small_covers row12281_block_005_covers

def row12281_large_left_right_right : List Nat := row12281_block_007 ++ row12281_block_008
theorem row12281_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12281 row12281_large_left_right_right :=
  Budget.RowUpperBound.branch row12281_block_007_covers row12281_block_008_covers

def row12281_large_left_right : List Nat := row12281_block_006 ++ row12281_large_left_right_right
theorem row12281_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12281 row12281_large_left_right :=
  Budget.RowUpperBound.branch row12281_block_006_covers row12281_large_left_right_right_covers

def row12281_large_left : List Nat := row12281_large_left_left ++ row12281_large_left_right
theorem row12281_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12281 row12281_large_left :=
  Budget.RowUpperBound.branch row12281_large_left_left_covers row12281_large_left_right_covers

def row12281_large_right_left : List Nat := row12281_block_009 ++ row12281_block_010
theorem row12281_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12281 row12281_large_right_left :=
  Budget.RowUpperBound.branch row12281_block_009_covers row12281_block_010_covers

def row12281_large_right_right_right : List Nat := row12281_block_012 ++ row12281_block_013
theorem row12281_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12281 row12281_large_right_right_right :=
  Budget.RowUpperBound.branch row12281_block_012_covers row12281_block_013_covers

def row12281_large_right_right : List Nat := row12281_block_011 ++ row12281_large_right_right_right
theorem row12281_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12281 row12281_large_right_right :=
  Budget.RowUpperBound.branch row12281_block_011_covers row12281_large_right_right_right_covers

def row12281_large_right : List Nat := row12281_large_right_left ++ row12281_large_right_right
theorem row12281_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12281 row12281_large_right :=
  Budget.RowUpperBound.branch row12281_large_right_left_covers row12281_large_right_right_covers

def row12281_large : List Nat := row12281_large_left ++ row12281_large_right
theorem row12281_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12281 row12281_large :=
  Budget.RowUpperBound.branch row12281_large_left_covers row12281_large_right_covers

def row12281 : List Nat := row12281_large
theorem row12281_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12281 row12281 := row12281_large_covers

def row12289_small : List Nat := []
theorem row12289_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12289 row12289_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12289_block_005 : List Nat := []
theorem row12289_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12289 row12289_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12289_block_006 : List Nat := []
def plan12289_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row12289_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12289 row12289_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12289_block_006
  · decide
  · rfl

def row12289_block_007 : List Nat := []
def plan12289_block_007 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 29), (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .second, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12289_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12289 row12289_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12289_block_007
  · decide
  · rfl

def row12289_block_008 : List Nat := []
theorem row12289_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12289 row12289_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12289_block_009 : List Nat := []
theorem row12289_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12289 row12289_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12289_block_010 : List Nat := []
theorem row12289_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12289 row12289_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12289_block_011 : List Nat := []
theorem row12289_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12289 row12289_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12289_block_012 : List Nat := []
theorem row12289_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12289 row12289_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12289_block_013 : List Nat := []
theorem row12289_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12289 row12289_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12289_large_left_left : List Nat := row12289_small ++ row12289_block_005
theorem row12289_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12289 row12289_large_left_left :=
  Budget.RowUpperBound.branch row12289_small_covers row12289_block_005_covers

def row12289_large_left_right_right : List Nat := row12289_block_007 ++ row12289_block_008
theorem row12289_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12289 row12289_large_left_right_right :=
  Budget.RowUpperBound.branch row12289_block_007_covers row12289_block_008_covers

def row12289_large_left_right : List Nat := row12289_block_006 ++ row12289_large_left_right_right
theorem row12289_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12289 row12289_large_left_right :=
  Budget.RowUpperBound.branch row12289_block_006_covers row12289_large_left_right_right_covers

def row12289_large_left : List Nat := row12289_large_left_left ++ row12289_large_left_right
theorem row12289_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12289 row12289_large_left :=
  Budget.RowUpperBound.branch row12289_large_left_left_covers row12289_large_left_right_covers

def row12289_large_right_left : List Nat := row12289_block_009 ++ row12289_block_010
theorem row12289_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12289 row12289_large_right_left :=
  Budget.RowUpperBound.branch row12289_block_009_covers row12289_block_010_covers

def row12289_large_right_right_right : List Nat := row12289_block_012 ++ row12289_block_013
theorem row12289_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12289 row12289_large_right_right_right :=
  Budget.RowUpperBound.branch row12289_block_012_covers row12289_block_013_covers

def row12289_large_right_right : List Nat := row12289_block_011 ++ row12289_large_right_right_right
theorem row12289_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12289 row12289_large_right_right :=
  Budget.RowUpperBound.branch row12289_block_011_covers row12289_large_right_right_right_covers

def row12289_large_right : List Nat := row12289_large_right_left ++ row12289_large_right_right
theorem row12289_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12289 row12289_large_right :=
  Budget.RowUpperBound.branch row12289_large_right_left_covers row12289_large_right_right_covers

def row12289_large : List Nat := row12289_large_left ++ row12289_large_right
theorem row12289_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12289 row12289_large :=
  Budget.RowUpperBound.branch row12289_large_left_covers row12289_large_right_covers

def row12289 : List Nat := row12289_large
theorem row12289_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12289 row12289 := row12289_large_covers

end ProjectEulerSolutions.PrimePairLarge
