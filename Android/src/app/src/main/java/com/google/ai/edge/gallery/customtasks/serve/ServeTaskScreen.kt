/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ai.edge.gallery.customtasks.serve

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel

@Composable
fun ServeTaskScreen(
  viewModel: ServeTaskViewModel = hiltViewModel(),
  modelManagerViewModel: ModelManagerViewModel = hiltViewModel(),
  task: Task? = null,
) {
  val uiState by viewModel.uiState.collectAsState()
  val context = LocalContext.current

  // Load the stored system prompt once when the screen is first shown.
  LaunchedEffect(task) {
    if (task != null) {
      viewModel.loadSystemPrompt(task)
    }
  }

  // Local editing state for the system prompt text field.
  var systemPromptDraft by remember(uiState.systemPrompt) {
    mutableStateOf(uiState.systemPrompt)
  }

  Column(
    modifier = Modifier
      .fillMaxSize()
      .padding(16.dp),
    horizontalAlignment = Alignment.CenterHorizontally
  ) {
    Text(
      text = "OpenAI Compatible Server",
      style = MaterialTheme.typography.headlineMedium
    )

    Spacer(modifier = Modifier.height(16.dp))

    Row(verticalAlignment = Alignment.CenterVertically) {
      Text("Status: ${if (uiState.isServerRunning) "Running" else "Stopped"}")
      Spacer(modifier = Modifier.width(16.dp))
      Button(onClick = {
        viewModel.toggleServer(context, modelManagerViewModel, 8080)
      }) {
        Text(if (uiState.isServerRunning) "Stop Server" else "Start Server")
      }
    }

    if (uiState.isServerRunning) {
      Spacer(modifier = Modifier.height(8.dp))
      Text("Listening on: ${uiState.serverAddress}")
      Spacer(modifier = Modifier.height(8.dp))
      Text("Model: ${modelManagerViewModel.uiState.collectAsState().value.selectedModel.name}")
    }

    Spacer(modifier = Modifier.height(16.dp))

    // System prompt section – allows configuring a default system instruction for the server.
    // When an API request includes its own system message that overrides this value.
    Text("System Prompt (used when API request has no system message):",
      style = MaterialTheme.typography.titleSmall)
    Spacer(modifier = Modifier.height(4.dp))
    OutlinedTextField(
      value = systemPromptDraft,
      onValueChange = { systemPromptDraft = it },
      modifier = Modifier
        .fillMaxWidth()
        .height(120.dp),
      placeholder = { Text("Optional system prompt…") },
      maxLines = 5,
    )
    Spacer(modifier = Modifier.height(4.dp))
    Button(
      onClick = { viewModel.saveSystemPrompt(systemPromptDraft) },
      enabled = systemPromptDraft != uiState.systemPrompt,
    ) {
      Text("Save System Prompt")
    }

    Spacer(modifier = Modifier.height(16.dp))
    Text("Logs:", style = MaterialTheme.typography.titleMedium)
    Spacer(modifier = Modifier.height(8.dp))

    LazyColumn(
      modifier = Modifier
        .fillMaxWidth()
        .weight(1f)
    ) {
      items(uiState.logs) { log ->
        Text(text = log, style = MaterialTheme.typography.bodySmall)
      }
    }
  }
}
