from django.contrib import admin
from .models import Dataset, DatasetColumn


class DatasetColumnInline(admin.TabularInline):
    model = DatasetColumn
    extra = 0

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        "dataset_id",
        "name",
        "user",
        "problem_type",
        "row_count",
        "column_count",
        "file_size_bytes",
        "is_archived",
        "created_at",
        "file_path",
    )
    list_filter = ("problem_type", "is_archived", "user")
    search_fields = ("name", "dataset_id")
    readonly_fields = ("dataset_id", "created_at")
    inlines = [DatasetColumnInline]
