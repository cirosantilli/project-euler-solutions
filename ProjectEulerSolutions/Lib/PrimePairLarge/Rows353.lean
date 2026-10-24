import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11807_small : List Nat := []
theorem row11807_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11807 row11807_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11807_block_005 : List Nat := []
theorem row11807_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11807 row11807_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11807_block_006 : List Nat := []
def plan11807_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row11807_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11807 row11807_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11807_block_006
  · decide
  · rfl

def row11807_block_007 : List Nat := [12647, 13619, 13757, 13799, 13907, 14159]
def plan11807_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 37), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .possible, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 37), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 23), (.firstFactor 13), .first, .possible, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .possible, (.secondFactor 17), .first, .first, .first, .first, .second, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 37), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11807_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11807 row11807_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11807_block_007
  · decide
  · rfl

def row11807_block_008 : List Nat := []
theorem row11807_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11807 row11807_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11807_block_009 : List Nat := []
theorem row11807_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11807 row11807_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11807_block_010 : List Nat := []
theorem row11807_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11807 row11807_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11807_block_011 : List Nat := []
theorem row11807_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11807 row11807_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11807_block_012 : List Nat := []
theorem row11807_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11807 row11807_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11807_block_013 : List Nat := []
theorem row11807_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11807 row11807_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11807_large_left_left : List Nat := row11807_small ++ row11807_block_005
theorem row11807_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11807 row11807_large_left_left :=
  Budget.RowUpperBound.branch row11807_small_covers row11807_block_005_covers

def row11807_large_left_right_right : List Nat := row11807_block_007 ++ row11807_block_008
theorem row11807_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11807 row11807_large_left_right_right :=
  Budget.RowUpperBound.branch row11807_block_007_covers row11807_block_008_covers

def row11807_large_left_right : List Nat := row11807_block_006 ++ row11807_large_left_right_right
theorem row11807_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11807 row11807_large_left_right :=
  Budget.RowUpperBound.branch row11807_block_006_covers row11807_large_left_right_right_covers

def row11807_large_left : List Nat := row11807_large_left_left ++ row11807_large_left_right
theorem row11807_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11807 row11807_large_left :=
  Budget.RowUpperBound.branch row11807_large_left_left_covers row11807_large_left_right_covers

def row11807_large_right_left : List Nat := row11807_block_009 ++ row11807_block_010
theorem row11807_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11807 row11807_large_right_left :=
  Budget.RowUpperBound.branch row11807_block_009_covers row11807_block_010_covers

def row11807_large_right_right_right : List Nat := row11807_block_012 ++ row11807_block_013
theorem row11807_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11807 row11807_large_right_right_right :=
  Budget.RowUpperBound.branch row11807_block_012_covers row11807_block_013_covers

def row11807_large_right_right : List Nat := row11807_block_011 ++ row11807_large_right_right_right
theorem row11807_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11807 row11807_large_right_right :=
  Budget.RowUpperBound.branch row11807_block_011_covers row11807_large_right_right_right_covers

def row11807_large_right : List Nat := row11807_large_right_left ++ row11807_large_right_right
theorem row11807_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11807 row11807_large_right :=
  Budget.RowUpperBound.branch row11807_large_right_left_covers row11807_large_right_right_covers

def row11807_large : List Nat := row11807_large_left ++ row11807_large_right
theorem row11807_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11807 row11807_large :=
  Budget.RowUpperBound.branch row11807_large_left_covers row11807_large_right_covers

def row11807 : List Nat := row11807_large
theorem row11807_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11807 row11807 := row11807_large_covers

def row11813_small : List Nat := []
theorem row11813_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11813 row11813_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11813_block_005 : List Nat := []
theorem row11813_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11813 row11813_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11813_block_006 : List Nat := [11927]
def plan11813_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .possible, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11813_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11813 row11813_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11813_block_006
  · decide
  · rfl

def row11813_block_007 : List Nat := [12659, 12953, 13151]
def plan11813_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 37), .possible, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .possible, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.secondFactor 37), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .second, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11813_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11813 row11813_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11813_block_007
  · decide
  · rfl

def row11813_block_008 : List Nat := []
theorem row11813_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11813 row11813_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11813_block_009 : List Nat := []
theorem row11813_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11813 row11813_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11813_block_010 : List Nat := []
theorem row11813_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11813 row11813_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11813_block_011 : List Nat := []
theorem row11813_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11813 row11813_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11813_block_012 : List Nat := []
theorem row11813_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11813 row11813_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11813_block_013 : List Nat := []
theorem row11813_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11813 row11813_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11813_large_left_left : List Nat := row11813_small ++ row11813_block_005
theorem row11813_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11813 row11813_large_left_left :=
  Budget.RowUpperBound.branch row11813_small_covers row11813_block_005_covers

def row11813_large_left_right_right : List Nat := row11813_block_007 ++ row11813_block_008
theorem row11813_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11813 row11813_large_left_right_right :=
  Budget.RowUpperBound.branch row11813_block_007_covers row11813_block_008_covers

def row11813_large_left_right : List Nat := row11813_block_006 ++ row11813_large_left_right_right
theorem row11813_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11813 row11813_large_left_right :=
  Budget.RowUpperBound.branch row11813_block_006_covers row11813_large_left_right_right_covers

def row11813_large_left : List Nat := row11813_large_left_left ++ row11813_large_left_right
theorem row11813_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11813 row11813_large_left :=
  Budget.RowUpperBound.branch row11813_large_left_left_covers row11813_large_left_right_covers

def row11813_large_right_left : List Nat := row11813_block_009 ++ row11813_block_010
theorem row11813_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11813 row11813_large_right_left :=
  Budget.RowUpperBound.branch row11813_block_009_covers row11813_block_010_covers

def row11813_large_right_right_right : List Nat := row11813_block_012 ++ row11813_block_013
theorem row11813_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11813 row11813_large_right_right_right :=
  Budget.RowUpperBound.branch row11813_block_012_covers row11813_block_013_covers

def row11813_large_right_right : List Nat := row11813_block_011 ++ row11813_large_right_right_right
theorem row11813_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11813 row11813_large_right_right :=
  Budget.RowUpperBound.branch row11813_block_011_covers row11813_large_right_right_right_covers

def row11813_large_right : List Nat := row11813_large_right_left ++ row11813_large_right_right
theorem row11813_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11813 row11813_large_right :=
  Budget.RowUpperBound.branch row11813_large_right_left_covers row11813_large_right_right_covers

def row11813_large : List Nat := row11813_large_left ++ row11813_large_right
theorem row11813_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11813 row11813_large :=
  Budget.RowUpperBound.branch row11813_large_left_covers row11813_large_right_covers

def row11813 : List Nat := row11813_large
theorem row11813_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11813 row11813 := row11813_large_covers

def row11821_small : List Nat := []
theorem row11821_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11821 row11821_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11821_block_005 : List Nat := []
theorem row11821_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11821 row11821_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11821_block_006 : List Nat := [12343]
def plan11821_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row11821_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11821 row11821_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11821_block_006
  · decide
  · rfl

def row11821_block_007 : List Nat := [13567, 13711, 14107, 14149]
def plan11821_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 7), .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, (.firstFactor 7), (.firstFactor 23), (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), (.firstFactor 17), .second, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.secondFactor 29), .first, .first, .possible, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .possible, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11821_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11821 row11821_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11821_block_007
  · decide
  · rfl

def row11821_block_008 : List Nat := []
theorem row11821_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11821 row11821_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11821_block_009 : List Nat := []
theorem row11821_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11821 row11821_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11821_block_010 : List Nat := []
theorem row11821_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11821 row11821_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11821_block_011 : List Nat := []
theorem row11821_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11821 row11821_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11821_block_012 : List Nat := []
theorem row11821_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11821 row11821_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11821_block_013 : List Nat := []
theorem row11821_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11821 row11821_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11821_large_left_left : List Nat := row11821_small ++ row11821_block_005
theorem row11821_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11821 row11821_large_left_left :=
  Budget.RowUpperBound.branch row11821_small_covers row11821_block_005_covers

def row11821_large_left_right_right : List Nat := row11821_block_007 ++ row11821_block_008
theorem row11821_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11821 row11821_large_left_right_right :=
  Budget.RowUpperBound.branch row11821_block_007_covers row11821_block_008_covers

def row11821_large_left_right : List Nat := row11821_block_006 ++ row11821_large_left_right_right
theorem row11821_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11821 row11821_large_left_right :=
  Budget.RowUpperBound.branch row11821_block_006_covers row11821_large_left_right_right_covers

def row11821_large_left : List Nat := row11821_large_left_left ++ row11821_large_left_right
theorem row11821_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11821 row11821_large_left :=
  Budget.RowUpperBound.branch row11821_large_left_left_covers row11821_large_left_right_covers

def row11821_large_right_left : List Nat := row11821_block_009 ++ row11821_block_010
theorem row11821_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11821 row11821_large_right_left :=
  Budget.RowUpperBound.branch row11821_block_009_covers row11821_block_010_covers

def row11821_large_right_right_right : List Nat := row11821_block_012 ++ row11821_block_013
theorem row11821_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11821 row11821_large_right_right_right :=
  Budget.RowUpperBound.branch row11821_block_012_covers row11821_block_013_covers

def row11821_large_right_right : List Nat := row11821_block_011 ++ row11821_large_right_right_right
theorem row11821_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11821 row11821_large_right_right :=
  Budget.RowUpperBound.branch row11821_block_011_covers row11821_large_right_right_right_covers

def row11821_large_right : List Nat := row11821_large_right_left ++ row11821_large_right_right
theorem row11821_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11821 row11821_large_right :=
  Budget.RowUpperBound.branch row11821_large_right_left_covers row11821_large_right_right_covers

def row11821_large : List Nat := row11821_large_left ++ row11821_large_right
theorem row11821_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11821 row11821_large :=
  Budget.RowUpperBound.branch row11821_large_left_covers row11821_large_right_covers

def row11821 : List Nat := row11821_large
theorem row11821_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11821 row11821 := row11821_large_covers

def row11827_small : List Nat := []
theorem row11827_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11827 row11827_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11827_block_005 : List Nat := []
theorem row11827_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11827 row11827_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11827_block_006 : List Nat := [12379]
def plan11827_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row11827_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11827 row11827_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11827_block_006
  · decide
  · rfl

def row11827_block_007 : List Nat := [12541, 12547, 12823]
def plan11827_block_007 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 37), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .possible, .possible, (.firstFactor 11), .first, .second, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 17), .second, .first, (.secondFactor 17), .first, .first, .possible, (.firstFactor 29), (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 29), (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11827_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11827 row11827_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11827_block_007
  · decide
  · rfl

def row11827_block_008 : List Nat := []
theorem row11827_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11827 row11827_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11827_block_009 : List Nat := []
theorem row11827_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11827 row11827_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11827_block_010 : List Nat := []
theorem row11827_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11827 row11827_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11827_block_011 : List Nat := []
theorem row11827_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11827 row11827_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11827_block_012 : List Nat := []
theorem row11827_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11827 row11827_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11827_block_013 : List Nat := []
theorem row11827_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11827 row11827_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11827_large_left_left : List Nat := row11827_small ++ row11827_block_005
theorem row11827_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11827 row11827_large_left_left :=
  Budget.RowUpperBound.branch row11827_small_covers row11827_block_005_covers

def row11827_large_left_right_right : List Nat := row11827_block_007 ++ row11827_block_008
theorem row11827_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11827 row11827_large_left_right_right :=
  Budget.RowUpperBound.branch row11827_block_007_covers row11827_block_008_covers

def row11827_large_left_right : List Nat := row11827_block_006 ++ row11827_large_left_right_right
theorem row11827_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11827 row11827_large_left_right :=
  Budget.RowUpperBound.branch row11827_block_006_covers row11827_large_left_right_right_covers

def row11827_large_left : List Nat := row11827_large_left_left ++ row11827_large_left_right
theorem row11827_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11827 row11827_large_left :=
  Budget.RowUpperBound.branch row11827_large_left_left_covers row11827_large_left_right_covers

def row11827_large_right_left : List Nat := row11827_block_009 ++ row11827_block_010
theorem row11827_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11827 row11827_large_right_left :=
  Budget.RowUpperBound.branch row11827_block_009_covers row11827_block_010_covers

def row11827_large_right_right_right : List Nat := row11827_block_012 ++ row11827_block_013
theorem row11827_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11827 row11827_large_right_right_right :=
  Budget.RowUpperBound.branch row11827_block_012_covers row11827_block_013_covers

def row11827_large_right_right : List Nat := row11827_block_011 ++ row11827_large_right_right_right
theorem row11827_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11827 row11827_large_right_right :=
  Budget.RowUpperBound.branch row11827_block_011_covers row11827_large_right_right_right_covers

def row11827_large_right : List Nat := row11827_large_right_left ++ row11827_large_right_right
theorem row11827_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11827 row11827_large_right :=
  Budget.RowUpperBound.branch row11827_large_right_left_covers row11827_large_right_right_covers

def row11827_large : List Nat := row11827_large_left ++ row11827_large_right
theorem row11827_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11827 row11827_large :=
  Budget.RowUpperBound.branch row11827_large_left_covers row11827_large_right_covers

def row11827 : List Nat := row11827_large
theorem row11827_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11827 row11827 := row11827_large_covers

end ProjectEulerSolutions.PrimePairLarge
